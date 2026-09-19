with open("verification/C015_laplacian_harness/01_test_laplacian_cell_coupling.py", "r") as f:
    code = f.read()

# Calibrar beta a 3.0 y pasos a 80
code = code.replace("BETA_HARNESS = 2.5", "BETA_HARNESS = 3.0")
code = code.replace("range(65)", "range(81)")
code = code.replace("[0, 8, 14, 15, 16, 20, 32, 48, 64]", "[0, 8, 14, 15, 16, 20, 32, 48, 64, 80]")

with open("verification/C015_laplacian_harness/01_test_laplacian_cell_coupling.py", "w") as f:
    f.write(code)

print("✓ Parámetros calibrados a física de convergencia (Beta=3.0, tau=80).")
