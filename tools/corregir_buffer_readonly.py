with open("verification/C007_spinor_lab/02_modal_coupling_real_data.py", "r") as f:
    code = f.read()

# Añadir .copy() para hacer el array mutable
old_read = 'patches = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, D)'
new_read = 'patches = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, D).copy()'

code = code.replace(old_read, new_read)

with open("verification/C007_spinor_lab/02_modal_coupling_real_data.py", "w") as f:
    f.write(code)

print("✓ Array marcado como mutable (.copy()).")
