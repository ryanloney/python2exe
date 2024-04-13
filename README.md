# python2exe
Turn a Python script into Windows executable app with CX Freeze

## Create and Activate Virtual Environment 
```
python -m venv exe_env
exe_env\Scripts\activate
```

## Clone the Repo
```
git clone https://github.com/ryanloney/python2exe
cd python2exe
```

## Install Dependencies 
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Create and Test Python Script
```
python run_openvino_demo.py
```

## Build EXE with CX Freeze 
`python setup.py build_exe`

NOTE: Make sure to include required files like OpenVINO IR by listing the setup.py file. 

See how to run the EXE demo here: https://gist.github.com/ryanloney#ai-pc-easy-button-demo 
