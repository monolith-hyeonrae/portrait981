# Enable sub-package discovery across multiple installed packages
import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)
