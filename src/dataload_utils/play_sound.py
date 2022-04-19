import sys
import getopt
import pyaudio  
import wave  

def myfunc(argv):
    arg_input = ""
    arg_output = ""
    arg_user = ""
    arg_help = "{0} -f <filename>".format(argv[0])
    
    try:
        opts, args = getopt.getopt(argv[1:], "hf:", ["filename"])
    except:
        print(arg_help)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-f", "--filename"): # Necessary
            arg_filename = arg

    return arg_filename
    
if __name__ == "__main__":
    FILENAME = myfunc(sys.argv)

    #define stream chunk   
    chunk = 1024  

    #open a wav format music  
    f = wave.open(FILENAME,"rb")  
    #instantiate PyAudio  
    p = pyaudio.PyAudio()  
    #open stream  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate = f.getframerate(),  
                    output = True)  
    #read data  
    data = f.readframes(chunk)  

    #play stream  
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  

    #stop stream  
    stream.stop_stream()  
    stream.close()  

    #close PyAudio  
    p.terminate()  