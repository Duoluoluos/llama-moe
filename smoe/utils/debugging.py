import socket
import torch
import debugpy
import torch.distributed as dist


def remote_breakpoint(host: str = "0.0.0.0", port: int = 5678, rank: int = 0):
    """
    This function helps to debug programs running in the remote computing node.

    In VSCode, you should add the configuration to the `.vscode/launch.json`, sth. like this 👇
    ```json
    {
        // Use IntelliSense to learn about possible attributes.
        // Hover to view descriptions of existing attributes.
        // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Remote Attach",
                "type": "python",
                "request": "attach",
                "connect": {
                    "host": "<hostname>",
                    "port": 5678
                },
                "pathMappings": [
                    {
                        "localRoot": "${workspaceFolder}",
                        "remoteRoot": "."
                    }
                ],
                "justMyCode": false
            }
        ]
    }
    ```

    Then, you could insert one line of code to the debugging position:
    ```python
    from smoe.utils.debugging import remote_breakpoint; remote_breakpoint()
    ```

    After the program starts and encounters the breakpoint, you could remote attach the debugger.
    """

    def _dp():
        print(
            f"Waiting for debugger to attach on {host}:{port}, server: {socket.gethostname()}..."
        )
        debugpy.listen((host, port))
        debugpy.wait_for_client()
        breakpoint()

    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == rank:
            _dp()
        dist.barrier()
    else:
        _dp()


def assert_finite(name: str, tensor: torch.Tensor):
    if not torch.isfinite(tensor).all():
        # 只打印/保存前几个异常元素，避免刷屏
        bad_mask = ~torch.isfinite(tensor)
        bad_vals = tensor[bad_mask][:8].detach().cpu()
        print(f"[NaN/Inf DETECTED] {name}: {bad_vals}")
        # 保存以便事后分析
        torch.save(tensor.detach().cpu(), f"/tmp/{name}_nan.pt")
        raise RuntimeError(f"Abort training: {name} contains NaN/Inf.")
    
def value_print(name, tensor: torch.Tensor):
    vals = tensor[:8].detach().cpu()
    print(f"[{name}] {vals}")

def cast_all_buffers(model, dtype, device=None):
    """
    递归把模型里所有 buffer（包括 inv_freq, cos_cached, sin_cached 等）
    原地转换到目标 dtype/device。
    不重新 register，直接改 _buffers 字典，规避“名字含点”问题。
    """
    for module in model.modules():                    # 遍历包含自身的所有子模块
        for buf_name, buf in module._buffers.items(): # 直接访问本模块 buffer
            if buf is not None:
                module._buffers[buf_name] = buf.to(dtype=dtype, device=device)

        # 特别处理 LlamaRotaryEmbedding 的缓存
        if hasattr(module, "max_seq_len_cached"):     # LlamaRotaryEmbedding 才有
            module.cos_cached = None
            module.sin_cached = None
            module.max_seq_len_cached = 0