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
```
python setup.py build_exe
```

## Test Windows Executable 
#### After the build is complete, you will see a directory called `build` inside the `python2exe` directory: 
![image](https://github.com/ryanloney/python2exe/assets/15709723/be6d797f-97a4-4917-b5ae-339203219f3e)


#### Open the `build` directory, then open `exe.win-amd64-3.11` (directory name may change based on your Python version): 
![image](https://github.com/ryanloney/python2exe/assets/15709723/a78082b1-75f2-448c-a5d5-2bcd59aa58be)


#### Double-click `run_openvino_demo.exe`: 
![image](https://github.com/ryanloney/python2exe/assets/15709723/dc797721-06f3-4d41-978e-1393f36ae2a9)


*See how to run a pre-built EXE demo here:* https://gist.github.com/ryanloney/412b27e1f4ef6cb40c2cc138703fcef5 
