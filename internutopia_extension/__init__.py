def import_extensions():
    import internutopia_extension.controllers
    import internutopia_extension.interactions
    import internutopia_extension.metrics
    import internutopia_extension.objects
    import internutopia_extension.robots
    import internutopia_extension.sensors
    import internutopia_extension.tasks


# Note: The extension module is located at internutopia_extension/envset/extension.py
# and is intended for Isaac Sim Omniverse Kit extension loading.
# It's not needed for standalone script execution.
try:
    from .envset.extension import *
except ImportError:
    # Running in standalone mode, extension not required
    pass