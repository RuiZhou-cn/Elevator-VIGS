"""A thin TensorRT execution wrapper for the engines scripts/build_trt_engines.sh writes.

`TrtRunner(engine_path)` deserializes one engine and runs it with the I/O tensor API: inputs
are torch CUDA tensors bound zero-copy by address, outputs are torch CUDA tensors the engine
writes into directly, and execution is enqueued on the current torch stream, so a caller can
consume the outputs in torch ops without a synchronization. Output buffers are reused across
calls while their shape and dtype hold.
"""
import numpy as np
import torch


class TrtRunner:
    def __init__(self, engine_path, input_names=None, output_names=None, logger_severity=None):
        import tensorrt as trt

        self.trt = trt
        if logger_severity is None:
            logger_severity = trt.Logger.ERROR
        self.logger = trt.Logger(logger_severity)

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        if not hasattr(self.engine, "num_io_tensors"):
            raise RuntimeError("This runner requires the TensorRT I/O tensor API (engine.num_io_tensors).")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context.")
        if not hasattr(self.context, "set_input_shape"):
            raise RuntimeError("context.set_input_shape not available; unexpected TensorRT version.")

        io_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        all_in = [n for n in io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        all_out = [n for n in io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if not all_in:
            raise RuntimeError("No INPUT tensors found in engine.")
        if not all_out:
            raise RuntimeError("No OUTPUT tensors found in engine.")

        # the caller may select and order a subset of the engine's tensors
        self.input_names = list(input_names) if input_names is not None else all_in
        self.output_names = list(output_names) if output_names is not None else all_out
        for n in self.input_names:
            if n not in all_in:
                raise ValueError(f"Requested input '{n}' not in engine inputs: {all_in}")
        for n in self.output_names:
            if n not in all_out:
                raise ValueError(f"Requested output '{n}' not in engine outputs: {all_out}")

        self._last_in_shapes = {}            # name -> shape tuple
        self._out_torch = {}                 # name -> torch.cuda tensor
        self._out_torch_shape = {}           # name -> shape tuple
        self._out_torch_dtype = {}           # name -> torch dtype

    def _np_dtype_to_torch(self, np_dtype):
        dt = np.dtype(np_dtype)
        m = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float16): torch.float16,
            np.dtype(np.int32): torch.int32,
            np.dtype(np.int8): torch.int8,
            np.dtype(np.uint8): torch.uint8,
            np.dtype(np.bool_): torch.bool,
        }
        if dt not in m:
            raise TypeError(f"Unsupported dtype for torch output: {dt}")
        return m[dt]

    def _bind_inputs(self, inputs):
        """Set the (dynamic) shape of every input and bind its address. `inputs` is a dict
        name -> torch CUDA tensor, or a list/tuple aligned with `input_names`."""
        trt = self.trt
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != len(self.input_names):
                raise ValueError(f"Got {len(inputs)} inputs, expected {len(self.input_names)}: {self.input_names}")
            in_dict = dict(zip(self.input_names, inputs))
        elif isinstance(inputs, dict):
            in_dict = inputs
        else:
            raise TypeError("inputs must be a dict{name: tensor} or list/tuple aligned with input_names")

        for name in self.input_names:
            if name not in in_dict:
                raise KeyError(f"Missing required input '{name}'. Required: {self.input_names}")
            x = in_dict[name]
            if not isinstance(x, torch.Tensor) or not x.is_cuda:
                raise TypeError(f"Input '{name}' must be a CUDA torch.Tensor.")
            if not x.is_contiguous():
                x = x.contiguous()          # a copy; pass contiguous tensors for zero-copy binding
            exp_torch = self._np_dtype_to_torch(trt.nptype(self.engine.get_tensor_dtype(name)))
            if x.dtype != exp_torch:
                raise TypeError(f"Input '{name}' dtype {x.dtype} != engine expects {exp_torch}")
            shape = tuple(x.shape)
            if self._last_in_shapes.get(name) != shape:
                self.context.set_input_shape(name, shape)
                self._last_in_shapes[name] = shape
            # zero-copy bind: TensorRT reads the torch tensor's memory
            self.context.set_tensor_address(name, int(x.data_ptr()))

    def _bind_outputs(self):
        """Allocate the torch CUDA outputs for the resolved output shapes and bind them."""
        trt = self.trt
        for out_name in self.output_names:
            out_shape = tuple(self.context.get_tensor_shape(out_name))
            if any(s < 0 for s in out_shape):
                raise RuntimeError(f"Dynamic output shape not resolved for '{out_name}': {out_shape}")
            out_torch_dtype = self._np_dtype_to_torch(trt.nptype(self.engine.get_tensor_dtype(out_name)))
            need = (
                (out_name not in self._out_torch) or
                (self._out_torch_shape.get(out_name) != out_shape) or
                (self._out_torch_dtype.get(out_name) != out_torch_dtype)
            )
            if need:
                self._out_torch[out_name] = torch.empty(out_shape, device="cuda", dtype=out_torch_dtype)
                self._out_torch_shape[out_name] = out_shape
                self._out_torch_dtype[out_name] = out_torch_dtype
            self.context.set_tensor_address(out_name, int(self._out_torch[out_name].data_ptr()))

    def run(self, inputs):
        """Enqueue one execution on the current torch stream; returns the output tensors in
        `output_names` order. Inputs produced on the current stream are ordered before it."""
        self._bind_inputs(inputs)
        self._bind_outputs()
        ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed (execute_async_v3 returned False).")
        return [self._out_torch[n] for n in self.output_names]
