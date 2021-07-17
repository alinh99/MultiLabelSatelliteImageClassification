Step 1: Download Python and download image dataset
Download Python on the website: https://www.python.org/downloads/ following the instruction.
Download image dataset in website: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data

Step 2: Install Libraries:

numpy

$ pip install numpy

torch, torchvision

Go to a website: https://pytorch.org/get-started/locally/ and follow the construction

Matplotlib

$ pip install matplotlib

Step 3: Run Python Project on Command Line
1.	Using the python Command:

To run Python scripts with the python command, you need to open a command-line and type in the word python, or python3 if you have both versions, followed by the path to your script, just like this:

$ python3 filename.py

If everything works okay, after you press Enter, you’ll see the output on your screen. That’s it! You’ve just run your first Python script!
If this doesn’t work right, maybe you’ll need to check your system PATH, your Python installation, the way you created the filename.py script, the place where you saved it, and so on.
This is the most basic and practical way to run Python scripts.


2.	Redirecting the Output:
Sometimes it’s useful to save the output of a script for later analysis. Here’s how you can do that:
$ python3 filename.py > output.txt

This operation redirects the output of your script to output.txt, rather than to the standard system output (stdout). The process is commonly known as stream redirection and is available on both Windows and Unix-like systems.
If output.txt doesn’t exist, then it’s automatically created. On the other hand, if the file already exists, then its contents will be replaced with the new output.
Finally, if you want to add the output of consecutive executions to the end of output.txt, then you must use two angle brackets (>>) instead of one, just like this:
$ python3 filename.py >> output.txt

Now, the output will be appended to the end of output.txt.

3.	Stop Python File:

To exit interactive mode, you can use one of the following options:
•	quit() or exit(), which are built-in functions.
•	The Ctrl+Z and Enter key combination on Windows, or just Ctrl+D on Unix-like systems.
If you’ve never worked with the command-line or terminal, then you can try this:
•	On Windows, the command-line is usually known as command prompt or MS-DOS console, and it is a program called cmd.exe. The path to this program can vary significantly from one system version to another.
A quick way to get access to it is by pressing the Win+R key combination, which will take you to the Run dialog. Once you’re there, type in cmd and press Enter.
•	On GNU/Linux (and other Unixes), there are several applications that give you access to the system command-line. Some of the most popular are xterm, Gnome Terminal, and Konsole. These are tools that run a shell or terminal like Bash, ksh, csh, and so on.
In this case, the path to these applications is much more varied and depends on the distribution and even on the desktop environment you use. So, you’ll need to read your system documentation.
•	On Mac OS X, you can access the system terminal from Applications → Utilities → Terminal.

4.	Run Python Scripts From an IDE or a Text Editor:
When developing larger and more complex applications, it is recommended that you use an integrated development environment (IDE) or an advanced text editor.
Most of these programs offer the possibility of running your scripts from inside the environment itself. It is common for them to include a Run or Build command, which is usually available from the tool bar or from the main menu.
Python’s standard distribution includes IDLE as the default IDE, and you can use it to write, debug, modify, and run your modules and scripts.
Other IDEs such as Eclipse-PyDev, PyCharm, Eric, and NetBeans also allow you to run Python scripts from inside the environment.
Advanced text editors like Sublime Text and Visual Studio Code also allow you to run your scripts.
To grasp the details of how to run Python scripts from your preferred IDE or editor, you can take a look at its documentation.

5.	Run Python Script from a File Manager:

Running a script by double-clicking on its icon in a file manager is another possible way to run your Python scripts. This option may not be widely used in the development stage, but it may be used when you release your code for production.
In order to be able to run your scripts with a double-click, you must satisfy some conditions that will depend on your operating system.
Windows, for example, associates the extensions .py and .pyw with the programs python.exe and pythonw.exe respectively. This allows you to run your scripts by double-clicking on them.
When you have a script with a command-line interface, it is likely that you only see the flash of a black window on your screen. To avoid this annoying situation, you can add a statement like input('Press Enter to Continue...') at the end of the script. This way, the program will stop until you press Enter.
This trick has its drawbacks, though. For example, if your script has any error, the execution will be aborted before reaching the input() statement, and you still won’t be able to see the result.
On Unix-like systems, you’ll probably be able to run your scripts by double-clicking on them in your file manager. To achieve this, your script must have execution permissions, and you’ll need to use the shebang trick you’ve already seen. Likewise, you may not see any results on screen when it comes to command-line interface scripts.
Because the execution of scripts through double-click has several limitations and depends on many factors (such as the operating system, the file manager, execution permissions, file associations), it is recommended that you see it as a viable option for scripts already debugged and ready to go into production.

References: https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/
            https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py