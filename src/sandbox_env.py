import random

def run_environment(kappa: float) -> dict:
    """
    Simulates a simple thermal dampener environment based on the kappa parameter.
    If kappa is too low (< 4.5), system is FREEZING.
    If kappa is too high (> 5.5), system is BOILING.
    If kappa is in target range (4.5 to 5.5), system is PERFECT.
    """
    # Base temperature driven by kappa (target is around 100 for PERFECT)
    base_temp = kappa * 20.0
    
    # Generate a short temperature log
    temp_log = []
    current_temp = base_temp - 5.0 # start a bit lower
    
    for _ in range(5):
        # random fluctuation
        current_temp += random.uniform(-2.0, 2.0)
        # dampening towards base_temp
        current_temp += (base_temp - current_temp) * 0.5
        temp_log.append(round(current_temp, 2))
        
    final_temp = temp_log[-1]
    
    # Determine system state and summary
    if kappa < 4.5:
        system_state = "FREEZING"
        summary = f"Kappa ({kappa}) is too low. System is freezing. Final temp: {final_temp}°C"
    elif kappa > 5.5:
        system_state = "BOILING"
        summary = f"Kappa ({kappa}) is too high. System is boiling. Final temp: {final_temp}°C"
    else:
        system_state = "PERFECT"
        summary = f"Kappa ({kappa}) is optimal. System is stable. Final temp: {final_temp}°C"
        
    return {
        "current_kappa": kappa,
        "temperature_log": temp_log,
        "final_temperature": final_temp,
        "system_state": system_state,
        "diagnostic_summary": summary
    }
