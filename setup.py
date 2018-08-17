from setuptools import setup, find_packages

setup(name='space_wrappers',
      version = '0.1.0',
      install_requires = ['gym'],
      test_rewuires = ["pytest"],
      packages = find_packages(),
      description = 'General purpose wrappers around OpenAI gym wrappers.',
      author = 'Erik Schultheis',
      author_email = 'erik.schultheis@stud.uni-goettingen.de',
      url = 'https://github.com/peterldowns/mypackage',  # use the URL to the github repo
)  
