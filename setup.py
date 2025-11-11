from setuptools import setup, find_packages

setup(
    name='MPNN',
    version='0.1',
    packages=find_packages(),
    author='Mikel M iparraguirre and Lucas Tesan',
    author_email='mikel.martinez@unizar.es',
    description='Library that contains GNNs to solve PDE problems',
    long_description_content_type='text/markdown',
    url='',
    python_requires='>=3.11',
    install_requires=[
        'matplotlib==3.8.3',
        'numpy==1.26.4',
        'torch==2.0.1=py3.11_cuda11.8_cudnn8_0',
        'pytorch-lightning==2.1.3',
        'torch-geometric==2.4.0',
        'torch-scatter==2.1.1',
        'torch-cluster==1.6.3',
        'wandb==0.16.2'
    ]
)