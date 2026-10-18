cd llama--win-vulkan-x64
.\llama-server.exe -m models/gemma-4-E2B_q4_0-it.gguf --mmproj models/gemma-4-E2B-it-mmproj.gguf -t 2 -c 16384 --host 0.0.0.0 --port 8080 --temp 0.7
pause