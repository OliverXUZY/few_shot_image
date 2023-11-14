import os
import argparse

parser = argparse.ArgumentParser("Training model")

parser.add_argument('--cpu', action='store_true')
parser.add_argument('--lianglab', action='store_true')
parser.add_argument('--wacc', action='store_true')
parser.add_argument('--research', action='store_true')

args = parser.parse_args()

read_file = "./script.sh"

f = open(read_file)
text = f.read()
f.close()
if args.wacc:
    text = text.replace("#SBATCH -p lianglab", "#SBATCH -p wacc")
    text = text.replace("device", "wacc")
    text = text.replace("#SBATCH -t 256:00:00          ## Walltime", "#SBATCH -t 1:00:00          ## Walltime")
elif args.research:
    text = text.replace("#SBATCH -p lianglab", "#SBATCH -p research")
    text = text.replace("device", "research")
elif args.cpu:
    text = text.replace("#SBATCH --gres=gpu:1          ## GPUs", "")
    text = text.replace("device", "cpu")
elif args.lianglab:
    text = text.replace("device", "lianglab")
else:
    text = text.replace("#SBATCH -p lianglab", "")
    text = text.replace("device", "none")

path = "./tmp.sh"

f = open(path, "w")
f.write(text)
f.close()

os.system("sbatch " + path)
os.remove(path)