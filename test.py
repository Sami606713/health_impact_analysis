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

print(get_packages("requirements.txt"))