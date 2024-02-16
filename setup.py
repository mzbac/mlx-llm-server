import setuptools
import pkg_resources

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

with open(this_directory / "requirements.txt") as fid:
    requirements = [str(r) for r in pkg_resources.parse_requirements(fid)]

setuptools.setup(
    name="mlx-llm-server",
    version="0.1.7",
    author="anchen",
    author_email="li.anchen.au@gmail.com",
    description="server to serve mlx model as an OpenAI compatible API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mzbac/mlx-llm",
    license="MIT",
    install_requires=requirements,
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "mlx-llm-server=mlx_llm_server.__main__:main",
        ],
    },
)
