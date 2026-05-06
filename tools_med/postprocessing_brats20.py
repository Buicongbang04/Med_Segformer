import os
import re
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
from skimage.morphology import remove_small_objects

