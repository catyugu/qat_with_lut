from gguf import GGUFReader

GGUF_PATH = "mlp_ternary.gguf"

try:
    print(f"--- Verifying GGUF file: {GGUF_PATH} ---")

    reader = GGUFReader(GGUF_PATH, 'r')

    print("\n[SUCCESS] GGUF file loaded without errors.")

    print("\n--- Metadata ---")
    for field in reader.fields.values():
        value = None
        if field.parts:
            value_part = field.parts[-1]
            if value_part.dtype.kind in ['S', 'U']:
                value = value_part.tobytes().decode(encoding='utf-8', errors='ignore')
            else:
                value = value_part[0]
        print(f"{field.name}: {value}")

    print("\n--- Tensors ---")
    for tensor in reader.tensors:
        print(f"Name: {tensor.name}, Shape: {tensor.shape}, Type: {tensor.tensor_type}, Size (bytes): {tensor.n_bytes}")

    print("\n[VERIFICATION PASSED] The GGUF file appears to be valid.")

except Exception as e:
    print(f"\n[VERIFICATION FAILED] An error occurred: {e}")