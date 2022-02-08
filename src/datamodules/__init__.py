from pathlib import Path
import importlib
from pytorch_lightning import LightningDataModule

# List all modules in the directory
modules = [p.stem for p in Path(__file__).parent.iterdir() if p.suffix == ".py" and p.stem != "__init__"]
to_expose = []

# Import all LightningDataModule's from the modules
for module_name in modules:
    module = importlib.import_module(f".{module_name}", "src.datamodules")
    to_expose.append((module_name, module))
    to_expose.extend([(name, getattr(module, name)) for name in dir(module) 
                        if isinstance(getattr(module, name), type) and
                        issubclass(getattr(module, name), LightningDataModule) and
                        name != "LightningDataModule"])

    
# Remove variables that are not supposed to be exposed
for var in dir():
    if not var.startswith("_") and var != "to_expose":
        del globals()[var]
 
# Update globals to enable from src.datamodueles import SSTDataModule 
# rather than from src.datamodueles.sst_datamodule import SSTDataModule
globals().update({name: obj for name, obj in to_expose})
del to_expose, var