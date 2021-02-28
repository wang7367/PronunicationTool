This is the prototype pronunciation tool. This readme will explain all the files 
that are included.

seniorDesign.py is the main program that runs the GUI via Tkinter. Note that on the record
screen, the play button will not do anything since we did not include any of our audio
databases. It would originally play back the reference audio for the word the user was attempting
to pronounce.

viet_deeplearning.py and mand_deeplearning.py predict the user's tone using the .h5 files
that are included. my_modelE15.h5 is for Mandarin Chinese and viet_my_modelE1000.h5 is
for Vietnamese.

spectrogram.py displays the pitch contour of the user's pronunciation and the reference
pronunciation. Since we did not include any audio databases, the reference side doesn't print
anything. 

requirements.txt contains the necessary python libraries for this program to run without error.
Do note that the 64-bit version of Python 3.7 was used. 32-bit Python will not work.




