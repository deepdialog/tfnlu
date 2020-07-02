
from .classification import main as cls_main
from .tagger import main as tag_main
from .parser import main as par_main


if __name__ == "__main__":
    cls_main()
    tag_main()
    par_main()
