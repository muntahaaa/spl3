import re
import subprocess

x1, y1, x2, y2 = (
    0.5391273498535156,
    0.8575849533081055,
    0.6944051384925842,
    0.9277631640434265
)

size_output = subprocess.check_output(
    ["adb", "shell", "wm", "size"],
    text=True
)

width, height = map(
    int,
    re.search(r"(\d+)x(\d+)", size_output).groups()
)

x = int(((x1 + x2) / 2) * width)
y = int(((y1 + y2) / 2) * height)

subprocess.run(
    ["adb", "shell", "input", "tap", str(x), str(y)],
    check=True
)