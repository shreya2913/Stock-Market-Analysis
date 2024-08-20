from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "Stock Market Analysis"
AUTHOR_USER_NAME = "shreya2913"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ['streamlit', 'pandas', 'matplotlib', 'prophet']


setup(
    name=SRC_REPO,
    version="0.0.7",
    author=AUTHOR_USER_NAME,
    description="A small package for Stock Market Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="shreyamudhigonda2002@gmail.com",
    packages=[SRC_REPO],
    python_requires=">=3.12",
    install_requires=LIST_OF_REQUIREMENTS
)