from setuptools import find_packages,setup

def get_packages(file_path:str):
    """
    This fun is responsible for installing the required packages

    input: str: str: path of the requirements.txt file
    output: list: list: list of required packages
    """
    try:
        with open(file_path,"r") as f:
            # read the packages
            requirements=f.read().splitlines()
            
            requirements=[req for req in requirements if req !="-e."]

            return requirements
    except Exception as e:
        return str(e)



setup(
    author="Samiullah",
    author_email="sami606713@gamil.com",
    name="Health Impact Analysis",
    description="This project project is totally dedicated to heath impact analysis",
    packages=find_packages(),  # Automatically find packages
    install_requires=get_packages("requirements.txt")
)