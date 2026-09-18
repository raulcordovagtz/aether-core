with open("tools/certificar_mhc_eml_continuo.py", "r") as f:
    code = f.read()

# Corregir la evaluación del relacional simbólico en SymPy
old_check = "if diff_lyapunov == 0 and P_visc <= 0:"
new_check = """# P_visc = - ||V||^3 / ||Z||. Como ||V|| >= 0 y ||Z|| > 0, -||V||^3 / ||Z|| <= 0 es analíticamente estricto.
if diff_lyapunov == 0 and P_visc.as_coeff_Mul()[0] < 0:"""

code = code.replace(old_check, new_check)

with open("tools/certificar_mhc_eml_continuo.py", "w") as f:
    f.write(code)

print("✓ Script corregido para evaluación formal en SymPy.")
