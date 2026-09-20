with open("speed_lab/pilot/19_test_native_metal_bridge.py", "r") as f:
    code = f.read()

# Corrección de obtención de puntero UMA
old_target_ptr = "u_target_ptr = ctypes.c_void_p(u_target.__array_interface__['data'][0])"
new_target_ptr = """u_target_np = np.array(u_target, copy=False)
u_target_ptr = ctypes.c_void_p(u_target_np.ctypes.data)"""

code = code.replace(old_target_ptr, new_target_ptr)

old_h_ptr = "h_ptr = ctypes.c_void_p(h_f32.__array_interface__['data'][0])"
new_h_ptr = """h_np = np.array(h_f32, copy=False)
    h_ptr = ctypes.c_void_p(h_np.ctypes.data)"""

code = code.replace(old_h_ptr, new_h_ptr)

with open("speed_lab/pilot/19_test_native_metal_bridge.py", "w") as f:
    f.write(code)

print("✓ Puntero ctypes corregido con zero-copy UMA.")
