""" Wrapper to TISEAN files.
"""

import tempfile
import subprocess
import os
from sys import platform as _platform
from time import strftime
import numpy as np
from collections import OrderedDict

__author__ = "Troels Bogeholm Mikkelsen"
__copyright__ = "Troels Bogeholm Mikkelsen 2016"
__credits__ = "Rainer Hegger, Holger Kantz and Thomas Schreiber"
__license__ = "MIT"
__version__ = "0.1"
__email__ = "bogeholm@nbi.ku.dk"
__status__ = "Development"

# Directory for temporary files
if "linux" in _platform:
    DIRSTR = '/tmp/'
elif _platform == "darwin":
    DIRSTR = '/private/tmp/'
else: # added this option so the code will work on non-Linux systems.  Need to look up the directory for other OS and add the appropriate conditionals. - Kyle L.
    Warning('Warning: Temp folder for this system not recognized.  Creating a folder in the current directory. Delete it when complete')
    DIRSTR = os.getcwd()
# Prefix to identify these files
PREFIXSTR = 'pytisean_temp_'
# suffix - TISEAN likes .dat
SUFFIXSTR = '.dat'

# We will use the present time as a part of the temporary file name
def strnow():
    """ Return 'now' as a string with hyphenation
    """
    return strftime('%Y-%m-%d-%H-%M-%S')

def genfilename():
    """ Generate a file name.
    """
    return PREFIXSTR + strnow() + '_'

def _gen_tmpFolder():
    """ generate a temporary folder
    """
    return tempfile.mkdtemp(prefix=genfilename(),
                            dir=DIRSTR)

def gentmpfile():
    """ Generate temporary file and return file handle.
    """
    fhandle = tempfile.mkstemp(prefix=genfilename(),
                               suffix=SUFFIXSTR,
                               dir=DIRSTR,
                               text=True)
    return fhandle

def _output_parser_remover(command, outFile_base, legacy=True):
    """ Parser for output
    Some tisean command, like d2, output multiple files, to handle this case,
    wrapper will find each file and load each into a numpy array.
    In legacy mode:
        * for single-file-output command, an array is returned;
        * for multiple-file-output command, a dictionary with output filetype
          as keyword is returned;
    In non-legacy mode:
        * Dictionary of entry format
            {keyword:output_np_array}
          will be returned.
        * For single-file-output command, keyword is "out";
        * For multiple-file-output command, keyword is filetype of each output
          file
    This routine also takes over the temporary output file's removal
    """
    output_data = OrderedDict();
    if command == 'd2':
        outFile_c2 = outFile_base+'.c2'
        outFile_d2 = outFile_base+'.d2'
        outFile_h2 = outFile_base+'.h2'
        try:
            output_data['c2'] = np.loadtxt(outFile_c2)
            output_data['d2'] = np.loadtxt(outFile_d2)
            output_data['h2'] = np.loadtxt(outFile_h2)
        finally:
            os.remove(outFile_c2)
            os.remove(outFile_d2)
            os.remove(outFile_h2)
    else:
        try:
            if legacy:
                output_data = np.loadtxt(outFile_base)
            else:
                output_data['out'] = np.loadtxt(outFile_base)
        finally:
            os.remove(outFile_base)
    return output_data

def tiseanio(command, *args, data=None, silent=False, legacy=True):
    """ TISEAN input/output wrapper.
        Accept numpy array 'data' - run 'command' on this and return result.
        This function is meant as a wrapper around the TISEAN package.
    """
    # Return values if 'command' (or something else) fails
    res = None
    err_string = 'Something failed!'

    workspace = _gen_tmpFolder()

    # If user specifies '-o' the save routine below will fail.
    if '-o' in args:
        raise ValueError('User is not allowed to specify an output file.')

    # Handles to temporary files
    if data is not None:
        fullname_in = os.path.join(workspace, "inFile")
    fullname_out = os.path.join(workspace, "outFile")

    # If no further args are specified, run this
    if not args:
        commandargs = [command, '-o', fullname_out]
    # Otherwise, we concatenate the args and command
    else:
        # User can specify float args - we convert
        arglist = [str(a) for a in args]
        if data is not None:
            commandargs = [command, fullname_in] + arglist + ['-o', fullname_out]
        else:
            commandargs = [command] + arglist + ['-o', fullname_out]

    # We will clean up irregardless of following success.
    try:
        # Save the input to the temporary 'in' file
        if data is not None:
            np.savetxt(fullname_in, data, delimiter='\t')

        # Here we call TISEAN (or something else?)
        subp = subprocess.Popen(commandargs,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        # Communicate with the subprocess
        (_, err_bytes) = subp.communicate()
        # Read the temporary 'out' file
        res = _output_parser_remover(command, fullname_out, legacy)
        # We will read this
        err_string = err_bytes.decode('utf-8')

    # Cleanup
    finally:
        if data is not None:
            os.remove(fullname_in)
        try:
            os.rmdir(workspace)  # all leftover files within will be removed
        except OSError:
            print("Additional non-data files were created")
            if not silent:
                print("\tNonsilent mode chosen, displaying additional content:\n")
            for remnant_file in os.listdir(workspace):
                if not silent:
                    print("File {} contains:".format(remnant_file))
                    with open(os.path.join(workspace, remnant_file)) as remnant_content:
                        print(remnant_content.read())
                os.remove(os.path.join(workspace, remnant_file))
            os.rmdir(workspace)

    if not silent:
        print(err_string)

    # We assume that the user wants the (error) message as well.
    return res, err_string

def tiseano(command, *args, silent=False, legacy=True):
    """ TISEAN output wrapper.
        Run 'command' and return result.
        This function is meant as a wrapper around the TISEAN package.
    """
    return tiseanio(command, *args, data=None,
                    silent=silent, legacy=legacy)
    
"""
Helper functions which pass the default values for each function into pytisean's tiseanio.
Documentation copied from the Tisean folder for ease of reference.

@author: Kyle Lemoi
"""

def ar_model(data,l=-1,x=0,m=1,c=1,p=5,s=0,V=1):
    """ Simple autoregressive (ar) model
    This program fits (by means of least squares) a simple autoregressive (AR) model to the possibly multivariate data. The model is given by the equation
    
    x_(n+1) = sum_(i=0)^(p-1) (A^i*x_(n-i) + noise)
    
    The matrices A^i are determined by the program.
    The output file contains the coefficients of the model and the residuals. Note that now the mean is subtracted first, conforming with common practice. 
    If you want to run the resulting model to generate a time series, you can either use the -s flag of the program or pipe the output to ar-run. 
    Note that no attempt has been made to generate a stable model.
    
    Input:
        
        Option	 Description                     Default
        data    Numpy array of data             N/A
        l       number of data to use	         whole file (-1 for len(data))
        x       number of lines to be ignored	0
        m       dimension of the vectors	      1
        c       column to be read	            1
        p       order of the model	            5
        s       iterate s steps of the model	   0 (no iteration)
        
        V	verbosity level
            0: only panic messages
            1: add input/output messages
            2: print residuals though iterating a model
    
    Output:
        
        The first line just contains the average forecast errors of the model for each component of the vector x, the next p*(dimension of the vectors) lines contain the AR coefficients for each of the components. 
        The rest of the file are the individual errors (residuals) or a iterated time series depending on the flag -s.
    """
    
    if (l == -1):
        l = len(data)

    if (s==0):
        acf, msg = tiseanio('ar-model', '-l',l, '-x',x,'-m',m,'-c',c,'-p',p,'-V',V,data=data)
    else:
        acf, msg = tiseanio('ar-model', '-l',l, '-x',x,'-m',m,'-c',c,'-p',p,'-s',s,'-V',V, data=data)
    
    return acf

def ar_run(data,l=1000,p=5,I=-1,x=10000,V=0):
    """Prints iterates of an autoregressive (AR) model
    
    Input:
        
        Option	 Description                     Default
        data    Numpy array of data             N/A
        l       number of iterations	         1000 (l=0 for infinite)
        p       order of the model	            5
        s       iterate s steps of the model	   0 (no iteration)
        x       number of transients discarded  10000
    
    Output:        
        
        The format is either one line containing the rms amplitude of the increments and one for each of the coefficients, or alternatively the output of ar-model.
    """
    acf, msg = tiseanio('ar-run','-l',l,'-p',p,'-I',I,'-x',x,'-V',V)
    
    return acf 

def arima_model(data,l=-1,x=0,m=1,c=1,p=10,P='0,0,0',I=50,e=0.001,s=-1,V=1):
    """This program fits (by means of least squares) an autoregressive integrated moving average (ARIMA) model to the possibly multivariate data. 
As a first step a AR model is fitted to give a first guess of the residuals which enter the MA part. With these residuals the full ARIMA is fitted. This fit is repeated until convergence of the residuals is reached or a maximum number of iterations was performed.
Note that no attempt has been made to generate a stable model.

    Input:
        
        Option  Description                         Default
        data    Numpy array of data                 N/A
        l       # of data to use                    -1 (whole file)
        x       # of lines to ignore                0
        m       dimension of vector                 1
        c       column to be read                   1
        p       order of the initial AR-model       10
        P       order of the AR,I,MA model          0,0,0 (just does the initial AR modeling)
        I       max # of iterations for ARIMA fit   50
        e       required accuracy for convergence   0.001
        s       length of the iterated data set     no iteration
        
        V:	     verbosity level
            
            0: only panic messages
            1: add input/output messages
            2: print residuals though iterating a model
            4: print original data + residuals
    
    Output:
        
        If P is not given:
        
            The first line contains the forecast error averaged over all components. 
            The second line contains the average forecast errors for all components individually. 
            The third line contains the Log-likelihood and the AIC values of the fit. 
            The next p*m lines contain the fitted ar coefficients and the rest of the file are either the residuals of the fit or a simulated new trajectory.
        
        If P is given:
            
            The first lines (marked #iteration xxx) show the convergence of the residuals of the ARIMA fit. 
            The rest of the files is like in the above case.
    """
    
    if (l == -1):
        l = len(data)

    if (s==0):
        acf, msg = tiseanio('arima-model', '-l',l, '-x',x,'-m',m,'-c',c,'-p',p,'-P',P,'-I',I,'-e',e,'-V',V,data=data)
    else:
        acf, msg = tiseanio('arima-model', '-l',l, '-x',x,'-m',m,'-c',c,'-p',p,'-P',P,'-I',I,'-e',e,'-s',s,'-V',V, data=data)
    
    return acf

def corr(data,l=-1,x=0,c=1,D=100,n=0,V=1): 
    """This program computes the autocorrelation of a scalar data set. 
    Depending on whether the -n flag is set or not, different normalizations are applied. With -n not set the definition is:
        
        C(tau) = 1/(N-tau) * (sum_i{(x_i-y)*(x_(i+tau)-y)})/s^2(x)
        
    where s is the standard deviation of the data and y its average.  Else it is
    
        C(tau) = 1/(N-tau)* sum_i{x_i*x_(i+tau)}
    
    Input:
        
        Option  Description                                     Default
        data    Numpy array of data                             N/A
        l       # of data to use                                -1 (whole file)
        x       # of lines to ignore                            0
        c       column to be read                               1
        D       number of correlations                          100
        n       don't use normalization to standard deviation   0 (normalize, 1 to turn off normalization)
        
        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages
    
    Output:
        
        The first two lines contain: 1. the average and 2. the standard deviation of the data. 
        The following lines are the autocorrelations (first column: delay, second column: autocorrelation).
    """
    
    
    if (l == -1):
        l = len(data)
    
    if (n==0):
        acf, msg = tiseanio('corr','-l',l,'-x',x,'-c',c,'-D',D,'-V',V,data=data)
    else:
        acf, msg = tiseanio('corr','-l',l,'-x',x,'-c',c,'-D',D,'-n','-V',V,data=data)
    
    return acf

def mem_spec(data,l=-1,x=0,c=1,p=128,P=2000,f=1,V=1):
    """This program estimates the power spectrum of a scalar data set on the basis of the maximum entropy principle. 
    A theoretical description of the idea behind is given in the Numerical Recipes.
    
    Input:
        
        Option  Description                                     Default
        data    Numpy array of data                             N/A
        l       # of data to use                                -1 (whole file)
        x       # of lines to ignore                            0
        c       column to be read                               1
        p       number of poles                                 128
        P       number of frequencies to print                  2000
        f       sampling rate in Hz                             1

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages
            2: add the AR coefficients to the output
    
    Output:
        
        The first line shows the average forecast error of the AR-model fitted. 
        The following p lines contain the coefficients of the fitted AR-model and the last P lines contain the power spectrum.
    """
    
    
    if (l == -1):
        l = len(data)
    
    acf, msg = tiseanio('mem_spec','-l',l,'-x',x,'-c',c,'-p',p,'-P',P,'-f',f,'-V',V,data=data)
    
    return acf

def spectrum(data,l=-1,x=0,c=1,f=1,w=-1, V=0):
    """Computes a power spectrum by binning adjacent frequencies according to the option -w.
    
    Input:
        
        Option  Description                                     Default
        data    Numpy array of data                             N/A
        l       # of data to use                                -1 (whole file)
        x       # of lines to ignore                            0
        c       column to be read                               1
        f       sampling rate (e.g. in Hz)                      1
        w       frequency resolution (e.g. in Hz)               -1 (for 1/N)

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages
    
    Output:
        
        Output file file_sp (frequency, spectrum).
    """
    
    
    if (l == -1):
        l = len(data)
    
    if (w ==-1):
        w = 1.0/l
    
    acf, msg = tiseanio('spectrum','-l',l,'-x',x,'-c',c,'-f',f,'-w',w,'-V',V,data=data)
    
    return acf        

def notch(data,X,l=-1,x=0,c=1,f=1,w=-1,V=0):
    """Simple notch filter in the time domain.
    
    Input:
        
        Option  Description                                     Default
        data    Numpy array of data                             N/A
        X       frequency to be canceled                        N/A
        l       # of data to use                                -1 (whole file)
        x       # of lines to ignore                            0
        c       column to be read                               1
        f       sampling rate of data                           1
        w       width of filter                                 -1 (f/100)


        V:      verbosity level
        
            0: only panic messages (default)
            1: add input/output messages
    
    Output:
        
    """
    
    
    if (l == -1):
        l = len(data)
    
    if (w ==-1):
        w = 1/f
        
    acf, msg = tiseanio('notch','-X',X,'-l',l,'-x',x,'-c',c,'-f',f,'-w',w,'-V',V,data=data)

    return acf        

def pca(data,l=-1,x=0,c=1,m='2,1',d=1,q=-1,W=0,V=0):
    """This program performs a global principal component analysis (PCA). 
    It gives the eigenvalues of the covariance matrix and depending on the -W flag eigenvectors, projections... of the input time series.
    
    Input:
        
        Option  Description                                     Default
        data    Numpy array of data                             N/A
        l       # of data to use                                -1 (whole file)
        x       # of lines to ignore                            0
        c       column to be read                               1
        m       no of input columns, embedding dimension        '2,1'
        d       delay                                           1
        q       definition depends on W                         -1 (full dimension)
        
        W:      desired output
        
            0: Just write the eigenvalues (default)
            1: Write the eigenvectors.  The columns of the output matrix are the eigenvectors.
            2: Transformation of the time series onto the eigenvector baiss.  the number of components printed is determined by the q flag.
            3: Project the time series onto the first -q eigenvectors

        V:      verbosity level
        
            0: only panic messages (default)
            1: add input/output messages
    
    Output:
        
        The output consists of the eigenvalues and depending on the -W flag of the eigenvectors, the new components or the projected time series.
    """
    
    
    if (l == -1):
        l = len(data)
    
    if (q ==-1):
        acf, msg = tiseanio('pca','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-W',W,'-V',V,data=data)
    else:
        acf, msg = tiseanio('pca','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-q',q,'-W',W,'-V',V,data=data)

    return acf      

def wiener1(data,l=-1,x=0,c=1,f=1,w=-1,V=0):
    """Wiener filter. 
    
    This first call produces the original periodogram. This may then be edited to provide the desired periodogram. 
    
    See also: wiener2
    
    Input:
        
        Option  Description                                     Default
        data    Numpy array of data                             N/A
        l       # of data to use                                -1 (whole file)
        x       # of lines to ignore                            0
        c       column to be read                               1
        f       sampling rate of data                           1
        w       frequency of resolution (e.g. in Hz)           -1 (for 1/N)


        V:      verbosity level
        
            0: only panic messages (default)
            1: add input/output messages        
        
    """
    
    
    if (l == -1):
        l = len(data)
    
    if (w ==-1):
        w = 1/f
        
    acf, msg = tiseanio('wiener1','-l',l,'-x',x,'-c',c,'-f',f,'-w',w,'-V',V,data=data)

    return acf     

def wiener2(data,l=-1,x=0,c=1,f=1,w=-1,V=0):
    """Wiener filter.
    
    This second call uses the output of wiener1 to generate a filtered sequence. 
    Internally, the series is padded with zeroes in order to get a FFT-able number of points. A warning is issued if applicable. 
    It is recommended to plot the spectral estimator computed by wiener1 in order to adjust the frequency resolution -w properly.
    
    See also: wiener1
    
    Input:
        
        Option  Description                                     Default
        data    Numpy array of data                             N/A
        l       # of data to use                                -1 (whole file)
        x       # of lines to ignore                            0
        c       column to be read                               1
        f       sampling rate of data                           1
        w       frequency of resolution (e.g. in Hz)           -1 (for 1/N)


        V:      verbosity level
        
            0: only panic messages (default)
            1: add input/output messages        
        
    """

    if (l == -1):
        l = len(data)
    
    if (w ==-1):
        w = 1/f
        
    acf, msg = tiseanio('wiener2','-l',l,'-x',x,'-c',c,'-f',f,'-w',w,'-V',V,data=data)

    return acf

def low121(data,l=-1,x=0,c=1,i=1,V=0):
    """This program applies a simple low pass filter in the time domain. 
    The filter works as follows:
        x'n=(xn-1+2xn+xn+1)/4
    
    Input:
        
        Option  Description                                     Default
        data    Numpy array of data                             N/A
        l       # of data to use                                -1 (whole file)
        x       # of lines to ignore                            0
        c       column to be read                               1
        i       sampling rate of data                           1

        V:      verbosity level
        
            0: only panic messages (default)
            1: add input/output messages
            2: print each iteration to a separate file
    
    Output:
        
        Depending on the the -V flag one or more files are produced. 
        Each of the files contains one column, consisting of the filtered time series.
        
    """

    if (l == -1):
        l = len(data)
        
    acf, msg = tiseanio('low121','-l',l,'-x',x,'-c',c,'-i',i,'-V',V,data=data)

    return acf

def sav_gol(data,l=-1,x=0,c=1,m=1,n='2,2',p=2,D=0,V=0):
    """This program performes a Savitzky-Golay filter to either clean the data from high frequency noise or to get a better estimate of the Dth derivative. It it possible to run it on multivariate data.
    
    Input:
        
        Option  Description                                     Default
        data    Numpy array of data                             N/A
        l       # of data to use                                -1 (whole file)
        x       # of lines to ignore                            0
        c       column to be read                               1
        m       # of components to be read (dimension)          1
        n       length of the averaging window back in time,    '2,2'
                length of the averaging window forward in time
        p       order of the fitted polynomial                  2
        D       order of the derivative to be estimated         0 (just filter)

        V:      verbosity level
        
            0: only panic messages (default)
            1: add input/output messages
    
    Output:
        
        The output file contains l lines, each of which has m columns. It is the filtered data. 
        The first length of the averaging window back in time and the last length of the averaging window forward in time lines are special. They contain the raw data in case that D was set to 0 and zeroes in case D was larger than zero.
        
    """

    if (l == -1):
        l = len(data)
        
    acf, msg = tiseanio('sav_gol','-l',l,'-x',x,'-c',c,'-m',m,'-n',n,'-p',p,'-D',D,'-V',V,data=data)

    return acf

def delay(data,l=-1,x=0,c=1,M=1,m=2,F=-1,d=1,D=-1,V=0):
    """This program produces delay vectors either from a scalar or from a multivariate time series.
    
    Input:
        
        Option  Description                                         Default
        data    Numpy array of data                                 N/A
        l       # of data to use                                    -1 (whole file)
        x       # of lines to ignore                                0
        c       column to be read                                   1
        M       Number of columns to be read                        1
        m       Embedding dimension                                 2
        F       The format for the embedding vector (see example)   -1 not set
        d       delay of the embedding vector                       1
        D       list of individual delays (see example)             -1 not set


        V:      verbosity level
        
            0: only panic messages (default)
            1: add input/output messages
    
    F flag example:
        
       Suppose a two component time series, with components xi and yi is given. From this series a 5 dimensional delay vector, which consists of 3 delayed x and 2 delayed y, shall be created. This would give delay vectors of the form (suppose the delay is 1)
       (xi,xi-1,xi-2,yi, yi-1)
       This is achieved by setting: -M2 -F 3,2, which means read 2 components (-M 2) and do the 3,2 (-F 3,2) embedding as shown above.
       It is also possible to combine the -M and -m flags. If the value of -m is an integer multiple of the -M value, the program assumes a symmetric embedding. E.g.: -M3 -m9 is the same as -M3 -F3,3,3. In case -m is not a integer multiple of -M the program complains and asks for -F.

        -F needs a list of comma separated "partial" dimensions. 
    
    D flag example:
        
        The idea of this flag is to allow the construction of delay vectors with varying delays. Suppose a delay vector of the form
        (x(i),x(i-d1),x(i-d1-d2))
        This is possible with the setting: -m3 -Dd1, d2
        -D needs a list of comma separated delays.
    
    Output:
        
        The delay vectors.
        
    """

    if (l == -1):
        l = len(data)
    
    if (F != -1 and D != -1):
        acf, msg = tiseanio('delay','-l',l,'-x',x,'-c',c,'-M',M,'-m',m,'-F',F,'-D',D,'-d',d,'-V',V,data=data)
    elif (F != -1):
        acf, msg = tiseanio('delay','-l',l,'-x',x,'-c',c,'-M',M,'-m',m,'-F',F,'-d',d,'-V',V,data=data)
    elif (D != -1):
        acf, msg = tiseanio('delay','-l',l,'-x',x,'-c',c,'-M',M,'-m',m,'-D',D,'-d',d,'-V',V,data=data)
    else: 
        acf, msg = tiseanio('delay','-l',l,'-x',x,'-c',c,'-M',M,'-m',m,'-d',d,'-V',V,data=data)
    
    return acf

def mutual(data,l=-1,x=0,c=1,b=16,D=20,V=0):
    """Estimates the time delayed mutual information of the data. It is the simplest possible realization. It uses a fixed mesh of boxes.
No finite sample corrections are implemented so far.

    Input:
        
        Option  Description                          Default
        data    Numpy array of data                  N/A
        l       # of data to use                     -1 (whole file)
        x       # of lines to ignore                 0
        c       column to be read                    1
        b       number of boxes for the partition    16
        D       maximal time delay                   20

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
        
    Output:
        
        The first line contains the number of occupied boxes, the second one the shannon entropy (normalized to the number of occupied boxes), the last D lines the mutual information (first column: delay, second column: mutual information).
        
    """

    if (l == -1):
        l = len(data)
    
    acf, msg = tiseanio('mutual','-l',l,'-x',x,'-c',c,'-b',b,'-D',D,'-V',V,data=data)
    
    return acf

def poincare(data,l=-1,x=0,c=1,m=2,d=1,q=-1,C=0,a=-1,V=0):
    """This programs makes a Poincaré section for time continuous scalar data sets along one of the coordinates of the embedding vector.
    
    Input:
        
        Option  Description                                     Default
        data    Numpy array of data                             N/A
        l       # of data to use                                -1 (whole file)
        x       # of lines to ignore                            0
        c       column to be read                               1
        m       embedding dimension                             2
        d       delay                                           1
        q       component for crossing                          -1 (the last)
        C       direction of the crossing (0: below, 1: above)  0
        a       position of the crossing                        average of the data

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
        
    Output:
        
        The output file contains m components for each cut. The first m-1 components are the coordinates of the vector at the crossing, the last one is the time between the last two crossings (see Hegger, Kantz).        
        
    """

    if (l == -1):
        l = len(data)
    
    if (q == -1):
        q = m
    
    if (a == -1):
        acf, msg = tiseanio('poincare','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-q',q,'-C',C,'-V',V,data=data)
    else: 
        acf, msg = tiseanio('poincare','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-q',q,'-C',C,'-a',a,'-V',V,data=data)
    return acf

def extrema(data,l=-1,x=0,c=1,m=1,w=1,z=0,t=0.0,V=0):
    """This program determines the maxima (minima) of one component of a possibly multivariate time series. This corresponds to a Poincaré section at the zeros of the derivative. To get a better estimate of the extremum, a quadratic interpolation is done.
    
    Input:
        
        Option  Description                                     Default
        data    Numpy array of data                             N/A
        l       # of data to use                                -1 (whole file)
        x       # of lines to ignore                            0
        c       column to be read                               1
        m       # of components                                 1
        w       which components to maxi(mini)mize              1
        z       determine minima instead of maxima              0 (maxima, 1 for minima)
        t       minimal time required between two extrema       0.0

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
        
    Output:
        
        The output consists of m+1 columns:
            First m columns: The position of the extrema
            Last column: The time between the last two extrema
        Note that the time given for the first extremum (the first row in the output file) is the absolute time from the start of the time series (t=0) to the first extremum.

    """

    if (l == -1):
        l = len(data)
        
    if (z == 0):
        acf, msg = tiseanio('extrema','-l',l,'-x',x,'-c',c,'-m',m,'-w',w,'-t',t,'-V',V,data=data)
    else: 
        acf, msg = tiseanio('extrema','-l',l,'-x',x,'-c',c,'-m',m,'-w',w,'-z','-t',t,'-V',V,data=data)
    return acf

def false_nearest(data,l=-1,x=0,c=1,m=1,M='1,5',d=1,f=2,t=0,V=3):
    """This program looks for the nearest neighbors of all data points in m dimensions and iterates these neighbors one step (more precisely delay steps) into the future. If the ratio of the distance of the iteration and that of the nearest neighbor exceeds a given threshold the point is marked as a wrong neighbor. The output is the fraction of false neighbors for the specified embedding dimensions (see Kennel et al.).
       Note: We implemented a new second criterion. If the distance to the nearest neighbor becomes smaller than the standard deviation of the data devided by the threshold, the point is omitted. This turns out to be a stricter criterion, but can show the effect that for increasing embedding dimensions the number of points which enter the statistics is so small, that the whole statistics is meanlingless. Be aware of this!

    Input:
        
        Option  Description                                                 Default
        data    Numpy array of data                                         N/A
        l       # of data to use                                            -1 (whole file)
        x       # of lines to ignore                                        0
        c       column to be read                                           1
        m       minimal embedding dimensions of the vectors                 1
        M       # of components, max. embedding dimension of the vectors    '1,5'
        d       delay of the vectors                                        1
        f       ratio factor                                                2.0
        t       theiler window                                              0

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
            2: add information about the current state of the program	
        
    Output:
        
        first column: the dimension (counted like shown above) 
        second column: the fraction of false nearest neighbors
        third column: the average size of the neighborhood
        fourth column: the average of the squared size of the neighborhood
    
    Details, Construction of the vectors:

        In case of a multivariate input the vectors are built in the following way (n= number of components):
            (x1(i),...,xn(i), x1(i+delay),...,xn(i+delay),..., x1(i+(maxemb-1)*delay),...,xn(i+(maxemb-1)*delay))
        The minimal embedding dimension given by the -m flag just refers to the embedding of the components. That means if you start with a three component vector and give -m1 -M3,3 -d1 then the program starts with 
            (x1(t),x2(t),x3(t))
        for the first test, then it takes
            (x1(t),x2(t),x3(t),x1(t+1), x2(t+1),x3(t+1),)
        for the next case and so forth.

    """

    if (l == -1):
        l = len(data)
        
    acf, msg = tiseanio('false_nearest','-l',l,'-x',x,'-c',c,'-m',m,'-M',M,'-d',d,'-f',f,'-t',t,'-V',V,data=data)
        
    return acf

def lzo_test(data,l=-1,x=0,c=1,m='1,2',d=1,n=-1,S=1,k=30,r=-1,f=1.2,s=1,C=-1,V=1):
    """This program makes a zeroth order ansatz and estimates the one step prediction errors of the model on a multivariate time series. This is done by searching for all neighbors of the point to be forecasted and taking as its image the average of the images of the neighbors. The given forecast errors are normalized to the standard deviations of each component. In addition to using a multicomponent time series a temporal embedding is possible. Thus one has to give two dimensions. The first is the number of components, the second the temporal embedding. This is realized by giving two numbers to the option m seperated by a comma.
Usage of the -c and -m flags

By default the first m columns of a file are used. This behaviour can by modified by means of the -c flag. It takes a series of numbers separated by commas. The numbers represent the colomns. For instance -m 3,1 -c 3,5,7 means, use three components with no additional temporal embedding, reading columns 3, 5 and 7. It is not necessary to give the full number of components to the -c flag. If numbers are missing, the string is filled up starting with the smallest number, larger than the largest given. For instance, -m 3,1 -c 3 would result in reading columns 3, 4 and 5.

    Input:
        
        Option  Description                                                 Default
        data    Numpy array of data                                         N/A
        l       # of data to use                                            -1 (whole file)
        x       # of lines to ignore                                        0
        c       column to be read                                           1
        m       number of components of the time series,embedding dimension     '1,2'
        d       delay for the embedding                                     1
        n       for how many points should the error be calculated          -1 (all points)
        S       temporal distance between the reference points              1
        k       minimal numbers of neighbors for the fit                    30
        r       neighborhood size to start with                             -1 (data size / 1000)
        f       factor to increase the neighborhood size by                 1.2
                if not enough neighbors were found
        s       steps to be forecased                                       1
        C       width of causality window                                   -1 (steps to be forecasted, s)

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
            2: add information about the current state of the program	
        
    Output:
        
        The output consists of s lines, each of which containing the steps forecasted (first column) and the relative forecast errors (next columns) for each component of the vector seperately. Relative means that the forecast error is devided by the standard deviation of the vector component.
        If the Verbosity level is larger than 1, the output also contains the individual forecast error for each component of each reference point. If s is larger than 1, the individual forecast errors are only given for the largest value of s.

    """

    if (l == -1):
        l = len(data)
    
    if (n == -1):
        n = len(data)
    
    if (r == -1):
        r = len(data)/1000
    
    if (C == -1):
        C = s
        
    acf, msg = tiseanio('lzo-test','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-n',n,'-S',S,'-k',k,'-r',r,'-f',f,'-s',s,'-C',C,'-V',V,data=data)
        
    return acf

def lzo_run(data,l=-1,x=0,c=1,m='1,2',d=1,L=1000,k=30,K=-1, p = 0, I = 0, r= -1, f= 1.2, V=1):
    """This program fits a locally zeroth order model to a possibly multivariate time series and iterates the time series into the future. The existing data set is extended starting with the last point in time. It is possible to add gaussian white dynamical noise during the iteration. In addition to using a multicomponent time series a temporal embedding is possible. Thus one has to give two dimensions. The first is the number of components, the second the temporal embedding. This is realized by giving two numbers to the option m seperated by a comma.
    Usage of the -c and -m flags

    By default the first m columns of a file are used. This behaviour can by modified by means of the -c flag. It takes a series of numbers separated by commas. The numbers represent the colomns. For instance -m 3,1 -c 3,5,7 means, use three components with no additional temporal embedding, reading columns 3, 5 and 7. It is not necessary to give the full number of components to the -c flag. If numbers are missing, the string is filled up starting with the smallest number, larger than the largest given. For instance, -m 3,1 -c 3 would result in reading columns 3, 4 and 5.
        
    Input:
        
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        m       number of components of the time series,embedding dimension     '1,2'
        d       delay for the embedding                                         1
        L       number of iterations into the future                            1000
        k       minimal numbers of neighbors for the fit                        30
        K       useo nly the nearest k neighbors found                          -1 (not set)
        p       add dynamical noise in units of percentage of the variance      0
        I       Seed for the random number generator used to add noise.         0 (time based seed)
                If set to 0 (zero) the time command is ued to create a seed.    
        k       minimal numbers of neighbors for the fit                        30
        r       neighborhood size to start with                                 -1 (data size / 1000)
        f       factor to increase the neighborhood size by                     1.2
                if not enough neighbors were found

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
            2: add information about the current state of the program	
        
    Output:
        
        The output consists of L lines, each of which contains a forecasted vector.
    """

    if (l == -1):
        l = len(data)
    
    if (r == -1):
        r = len(data)/1000
    
    if (K == -1):
        acf, msg = tiseanio('lzo-run','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-L',L,'-k',k,'-%',p,'-I',I,'-r',r,'-f',f,'-V',V,data=data)
    else:
        acf, msg = tiseanio('lzo-run','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-L',L,'-k',k,'-K',K,'-%',p,'-I',I,'-r',r,'-f',f,'-V',V,data=data)
        
    return acf

def lfo_test(data,l=-1,x=0,c=1,m='1,2',d=1,n=-1,k=30,r=-1, f= 1.2, s = 1, C=-1, V=1):
    """This program makes a local linear ansatz and estimates the one step prediction error of the model. It allows to determine the optimal set of parameters for the program lfo-run, which iterates the local linear model to get a clean trajectory. The given forecast error is normalized to the variance of the data.
    
    Input:
        
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        m       number of components of the time series,embedding dimension     '1,2'
        d       delay for the embedding                                         1
        n       for how many points hsould the eror be calculated               -1 (all)
        k       minimal numbers of neighbors to fit                             30
        r       neighborhood size to start with                                 -1 (data interval/ 1000)
        f       factor to increase the neighborhood size by                     1.2 
                if not enough neighbors were found 
        s       steps to be forecasted                                          1
        C       width of causality window                                       -1 (steps to be forecasted, s)

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
            2: add information about the current state of the program	
        
    Output:
        
        For -V<2 the output consists of (# of components) numbers only, the relative forecast errors. Relative means, the forecast errors are devided by the standard deviations of the components.
        For -V>=2 the individual forecast errors are given, too.
    """

    if (l == -1):
        l = len(data)
    
    if (n == -1):
        n = len(data)
    
    if (r == -1):
        r = len(data)/1000
        
    if (C == -1):
        C = s
    
    acf, msg = tiseanio('lfo-test','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-n',n,'-k',k,'-r',r,'-f',f,'-s',s,'-C',C,'-V',V,data=data)
        
    return acf

def lfo_run(data,l=-1,x=0,c=1,m='1,2',d=1,L=1000,k=30,r=-1, f= 1.2,O=0, V=1):
    """This program makes depending on whether -0 is set either a local linear ansatz or a zeroth order ansatz for a possibly multivariate time series and iterates a artificial trajectory. The initial values for the trajectory are the last points of the original time series. Thus it actually forecasts the time series.
    
    Input:
        
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        m       number of components of the time series,embedding dimension     '1,2'
        d       delay for the embedding                                         1
        L       length of prediction                                            1000
        k       minimal numbers of neighbors to fit                             30
        r       neighborhood size to start with                                 -1 (data interval/ 1000)
        f       factor to increase the neighborhood size by                     1.2 
                if not enough neighbors were found 
        O       perform a zeroth order fit instead of a local linear one        0 (local linear, 1 for zeroth order)

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
            2: add information about the current state of the program	
        
    Output:
        
        The output is just the components of the forecasted time series.
    """

    if (l == -1):
        l = len(data)
    
    if (r == -1):
        r = len(data)/1000
    
    if (O == 0):
        acf, msg = tiseanio('lfo-run','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-L',L,'-k',k,'-r',r,'-f',f,'-V',V,data=data)
    else: 
        acf, msg = tiseanio('lfo-run','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-L',L,'-k',k,'-r',r,'-f',f,'-O','-V',V,data=data)
        
    return acf

def lfo_ar(data, l=-1, x=0, c=1, m='1,2', d=1, i=-1, r=-1, R=-1, f= 1.2, s = 1, C=-1, V=1):
    """This program makes a local linear ansatz and estimates the one step prediction error of the model. The difference to lfo-test is that it does it as a function of the neighborhood size (see Casdagli).
    lfo-ar means something like local-first-order -› AR-model
    
    
    Input:
        
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        m       number of components of the time series,embedding dimension     '1,2'
        d       delay for the embedding                                         1
        n       for how many points hsould the eror be calculated               -1 (all)
        r       neighborhood size to start with                                 -1 (data interval/ 1000)
        R       neighborhood size to end with                                   -1 (data interval)
        f       factor to increase the neighborhood size by                     1.2 
                if not enough neighbors were found 
        s       steps to be forecasted                                          1
        C       width of causality window                                       -1 (steps to be forecasted, s)

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
            2: add information about the current state of the program	
        
    Output:
        
        The output consists of 5 columns for each neighborhood size:
            neighborhood size (units of the data)
            relative forecast error ((forecast error)/(variance of the data))
            fraction of points for which neighbors were found for this neighborhood size
            average number of neighbors found per point
            variance of the fraction of points for which neighbors were found
    """

    if (l == -1):
        l = len(data)
    
    if (i == -1):
        i = len(data)
    
    if (r == -1):
        r = len(data)/1000
    
    if (R == -1):
        r = len(data)
    
    if (C == -1):
        C = s
    
    acf, msg = tiseanio('lfo-ar','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-i',i,'-r',r,'-R',R,'-f',f,'-s',s,'-C',C,'-V',V,data=data)
        
    return acf

def lzo_gm(data, l=-1, x=0, c=1, m='1,2', d=1, i=-1, r=-1, R=-1, f= 1.2, s = 1, C=-1, V=1):
    """This program performs a constant (zeroth order) fit as a function of the neighborhood size.
    lzo-gm means something like local-zeroth-order -› global mean
    
    
    Input:
        
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        m       number of components of the time series,embedding dimension     '1,2'
        d       delay for the embedding                                         1
        n       for how many points hsould the eror be calculated               -1 (all)
        r       neighborhood size to start with                                 -1 (data interval/ 1000)
        R       neighborhood size to end with                                   -1 (data interval)
        f       factor to increase the neighborhood size by                     1.2 
                if not enough neighbors were found 
        s       steps to be forecasted                                          1
        C       width of causality window                                       -1 (steps to be forecasted, s)

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
            2: add information about the current state of the program	
        
    Output:
        The output consists of dimension+4 columns for each neighborhood size:
            neighborhood size (units of the data)
            relative forecast error ((forecast error)/(variance of the data))
            relative forecast errors for the individual components of the vector
            fraction of points for which neighbors were found for this neighborhood size
            average number of neighbors found per point
            variance of the fraction of points for which neighbors were found
    """

    if (l == -1):
        l = len(data)
    
    if (i == -1):
        i = len(data)
    
    if (r == -1):
        r = len(data)/1000
    
    if (R == -1):
        r = len(data)
    
    if (C == -1):
        C = s
    
    acf, msg = tiseanio('lfo-gm','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-i',i,'-r',r,'-R',R,'-f',f,'-s',s,'-C',C,'-V',V,data=data)
        
    return acf

def rbf(data, L, l=-1, x=0, c=1, m=2, d=1, p=10, X=0, s=1, n= -1, V=1):
    """This program models the data using a radial basis function (rbf) ansatz. The basis functions used are gaussians, with center points chosen to be data from the time series. 
    If the -X option is not given, a kind of Coulomb force is applied to them to let them drift a bit in order to distribute them more uniformly. 
    The variance of the gaussians is set to the average distance between the centers.
    This program either tests the ansatz by calculating the average forecast error of the model, or makes a i-step prediction using the -L flag, additionally. The ansatz made is:
        x_(n+1)=a_0+SUM a_i f_i(x_n),
    where x_n is the nth delay vector and f_i is a gaussian centered at the ith center point.
    
    Input:
        
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        L       determines the length of the predicted series                   N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        m       embedding dimension                                             2
        d       delay                                                           1
        p       number of centers                                               10
        X       deactivate drift (Coulomb Force)                                0
        s       steps to forecast (for the forecast error)                      1
        n       number of points for the fit                                    -1 (number of data points)

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
            2: add information about the current state of the program	
        
    Output:
        The output file contains: The coordinates of the center points, the variance used for the gaussians, the coefficients (weights) of the basis functions used for the model, the forecast errors and if the -L flag was set, the predicted points.
    """

    if (l == -1):
        l = len(data)
    
    if (n == -1):
        n = len(data)
    
    if (X == 0):
            acf, msg = tiseanio('rbf','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-p',p,'-s',s,'-n',n,'-L',L,'-V',V,data=data)
    else:
            acf, msg = tiseanio('rbf','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-p',p,'-X','-s',s,'-n',n,'-L',L,'-V',V,data=data)
        
    return acf

def polynom(data, L, l=-1, x=0, c=1, m=2, d=1, p=2, n= -1, V=1):
    """This programs models the data making a polynomial ansatz.
    
    Input:
        
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        L       determines the length of the predicted series                   N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        m       embedding dimension                                             2
        d       delay                                                           1
        p       order of the polynomial                                         2
        n       number of points for the fit                                    -1 (number of data points)
                The other points are used to estimate the out of sample error

        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
            2: add information about the current state of the program	
        
    Output:
        The output file contains: The coordinates of the center points, the variance used for the gaussians, the coefficients (weights) of the basis functions used for the model, the forecast errors and if the -L flag was set, the predicted points.
    """

    if (l == -1):
        l = len(data)
    
    if (n == -1):
        n = len(data)
    
    acf, msg = tiseanio('polynom','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-p',p,'-n',n,'-L',L,'-V',V,data=data)

    return acf

def d2(data, l=-1, x=0, c=1, M='1,10', d=1, t=0, R=-1,r=-1,ep=100,N =1000,E=0, V=1):
    """This program estimates the correlation sum, the correlation dimension and the correlation entropy of a given, possibly multivariate, data set. It uses the box assisted search algorithm and is quite fast as long as one is not interested in large length scales. All length scales are computed simultaneously and the output is written every 2 min (real time, not cpu time). It is possible to set a maximum number of pairs. If this number is reached for a given length scale, the length scale will no longer be treated for the rest of the estimate.
    
    Please consult the introduction paper for initial material on dimension estimation. If you are serious, you will need to study some of the literature cited there as well.
    
    In extension to what is described there, the simultaneous use of multivariate data and temporal embedding is possible using d2. Thus one has to give two numbers in order to specify the embedding dimension. The first is the number of multivariate components, the second the number of lags in the temporal embedding. This is realized by giving two numbers to the option -M, seperated by a comma. For a standard scalar time series, set the first number to 1. If your multivariate data spans the whole phase space and no further temporal embedding is desired, set the second value to 1. In any case, the total embedding dimension in the sense of the embedding theorems is the product of the two numbers.
    
    In order to be able to assess the convergence with increasing embedding dimension, results are reported for several such values. The inner loop steps through the number of components until the first argument of M is reached. The outer loop increases the number of time lags until the second argument of M is reached.
    
    Input:
        
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        M       # of components                                                 '1,10'
                maximal embedding dimension
        d       delay of the delay vectors                                      1
        t       theiler window                                                  0
        R       maximal length scale                                            -1 (max data interval)
        r       minimal length scale                                            -1 (max data interval / 1000)
        ep      number of epsilon values                                        100
        N       maximal number of pairs to be used                              1000
        E       use data that is normalized to [0,1] for all components         0 (use natural units of the data, 1 for normalization)
        
        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages (default)
            2: add information about the current state of the program	
            
        Usage of the -c and -M flags

            Suppose, the option -M x,y has been specified. By default, the first x columns of a file are used. This behaviour can be modified by means of the -c flag. It takes a series of numbers separated by commas. The numbers represent the colomns. For instance -M 3,1 -c 3,5,7 means, use three components with no additional temporal embedding, reading columns 3, 5 and 7. It is not necessary to give the full number of components to the -c flag. If numbers are missing, the string is filled up starting with the smallest number, larger than the largest given. For instance, -M 3,1 -c 3 would result in reading columns 3, 4 and 5.
    
    Output:
        
        Returns a dict with three entries.  Each dict entry has two columns.
        
        first column: epsilon (in units chosen)
        
        second column: the estimated quantity
            
            c2: the correlations sums for all treated length scales and embedding dimensions
            d2: the local slopes of the logarithm of the correlation sum, the correlation dimension
            h2: the correlation entropies
    """
    
    if (l == -1):
        l = len(data)
    
    if (R ==-1):
        R = len(data)
        
    if (r ==-1):
        r = len(data)/1000
    
    if (E == 0):
        acf, msg = tiseanio('d2','-l',l,'-x',x,'-c',c,'-M',M,'-d',d,'-t',t,'-R',R,'-r',r,'-#',ep,'-N',N,'-V',V,data=data)
    else:
        acf, msg = tiseanio('d2','-l',l,'-x',x,'-c',c,'-M',M,'-d',d,'-t',t,'-R',R,'-r',r,'-#',ep,'-N',N,'-E','-V',V,data=data)
        
    return acf

def c1(data, d,m,M,t,n, l=-1, x=0, c=1,res = 2,K=100,V=0):
    
    """Information Dimension
    
    Computes curves for the fixed mass computation of the information dimension. The output is written to a file named file_c1, containing as two columns the necessary radius and the `mass'. Although the `mass' is the independent quantity here, this is to conform with the output of c2naive and d2.
    A logarithmic range of masses between 1/N and 1 is realised by varying the neighbour order k as well as the subsequence length n. For a given mass k/n, n is chosen as small is possible as long as k is not smaller than the value specified by -K .

    The number of reference points has to be selected by specifying -n . That number of points are selected at random from all time indices.

    It is possible to use multivariate data, also with mixed embeddings. Contrary to the convention, the embedding dimension here specifies the total number of phase space coordinates. The number of components of the time series to be considered can only be given by explicit enumeration with the option -c .

    Note: You will probably use the auxiliary programs c2d or c2t to process the output further. The formula used for the Gaussian kernel correlation sum does not apply to the information dimension.
    
    Input:
        
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        m       minimal embedding dimension (int)                               N/A
        M       maximal embedding dimension (int)                               N/A
        t       minimal time separation                                         N/A
        n       minimal number of center point                                  N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        res     resolution, values per octave                                   2
        K       maximal number of neighbors        
        
        V:      verbosity level
        
            0: only panic messages (default)
            1: add input/output messages 
            2: add information about the current state of the program	
            
    """
    
    if (l == -1):
        l = len(data)
    

    acf, msg = tiseanio('c1','-l',l,'-x',x,'-c',c,'-d',d,'-m',m,'-M',M,'-t',t,'-n',n,'-#',res,'-K',K,'-V',V,data=data)
        
    return acf

def boxcount(data,l=-1,x=0,c=1,d=1,M='1,10',Q=2,R=-1,r=-1,ep=20,V=1):
    
    """This program estimates the Renyi entopy of Qth order using a partition of the phase space instead of using the Grassberger-Procaccia scheme. 
    The program also can handle multivariate data, so that the phase space is build of the components of the time series plus a temporal embedding, if desired. 
    I should mention that the memory requirement does not increase exponentially like 1/epsilonM but only like M*(length of series). So it can also be used for small epsilon and large M.
    No finite sample corrections are implemented so far.
    
    Input:
    
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        d       delay for the delay vectors                                     1
        M       # of components, maximal embedding dimension                    '1,10'
        Q       order of the entropy                                            2
        R       maximal length scale                                            -1 (data range)
        r       minimal length scale                                            -1 (data range / 1000)
        ep      number of epsilon values                                               
        
        V:      verbosity level
        
            0: only panic messages (default)
            1: add input/output messages 
            2: add information about the current state of the program	
        
    Output:
        
        three columns for each dimension and for each epsilon value:
            1)epsilon
            2) Qth order entropy (HQ(dimension,epsilon))
            3) Qth order differential entropy ( HQ(dimension,epsilon)-HQ(dimension-1,epsilon))

        The slope of the second line gives an estimate of DQ(m,epsilon).
    """
    
    if (l ==-1):
        l = len(data)
        
    if (R == -1):
        R = len(data)
        
    if (r == -1):
        r = len(data)/1000
    
    acf, msg = tiseanio('boxcount','-l',l,'-x',x,'-c',c,'-d',d,'-M',M,'-Q',Q,'-R',R,'-r',r,'-#',ep,'-V',V,data=data)
    
    return acf

def lyap_k(data,l=-1,x=0,c=1,M=2,m=2,d=1,r=-1,R=-1,L=5,n=-1,s=50,t=0,V=3):
    """The program estimates the largest Lyapunov exponent of a given scalar data set using the algorithm of Kantz.
    
    Input:
    
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        M       maximal embedding dimension to use                              2
        m       minimal embedding dimension to use                              2
        d       delay to use                                                    1
        R       maximal length scale to search neighbors                        -1 (data range)
        r       minimal length scale to search neighbors                        -1 (data range / 1000)
        L       number of length scales to use                                  5
        n       number of reference points to use                               -1 (all)
        s       number of iterations in time                                    50
        t       theiler window                                                  0                                               
        
        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages 
            2: add information about the current state of the program
        
    Output:
        
        For each embedding dimension and each length scale the output contains a block of data consisting of 3 columns:
            1) The number of the iteration
            2) The logarithm of the stretching factor (the slope is the Lyapunov exponent if it is a straight line)
            3) The number of points for which a neighborhood with enough points was found
    
    """
    
    if (l == -1):
        l = len(data)
    
    if (R == -1):
        R = len(data)/100
        
    if (r == -1):
        r = len(data)/1000
    
    if (n == -1):
        n = len(data)
    
    acf, msg = tiseanio('lyap_k','-l',l,'-x',x,'-M',M,'-m',m,'-d',d,'-r',r,'-R',R,'-#',L,'-n',n,'-s',s,'-t',t,'-V',V,data=data)
    
    return acf

def lyap_r(data,l=-1,x=0,c=1,m=2,d=1,t=0,r=-1,s=50,V=3):
    """The program estimates the largest Lyapunov exponent of a given scalar data set using the algorithm of Rosenstein et al.
    
    Input:
        
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        m       embedding dimension to use                                      2
        d       delay to use                                                    1
        t       theiler window                                                  0
        r       minimal length scale for the neighborhood search                -1 (data interval / 1000)
        s       number of iteraions in time                                     50
    
        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages 
            2: add information about the current state of the program
    
    Output: 
        
        Two column array: 
            First column: Number of the iteration
            Second column: Logarithm of the stretching factor
    """
    
    if (l==-1):
        l= len(data)
    
    if (r==-1):
        r = len(data)/1000
    
    acf, msg = tiseanio('lyap_r','-l',l,'-x',x,'-c',c,'-m',m,'-d',d,'-t',t,'-r',r,'-s',s,'-V',V,data=data)
    
    return acf

def lyap_spec(data,l=-1,x=0,c=1,m='1,2',r=-1,f=1.2,k=30,n=-1,I=0,V=1):
    
    """This program estimates the whole spectrum of Lyapunov exponents for a given, possibly multivariate, time series. Whole spectrum means: If d components are given and the embedding dimension is m than m*d exponents will be determined. The method is based on the work of Sano and Sawada.
    
    Input:
        
        Option  Description                                                     Default
        data    Numpy array of data                                             N/A
        l       # of data to use                                                -1 (whole file)
        x       # of lines to ignore                                            0
        c       column to be read                                               1
        m       no. of components, embedding dimension                          '1,2'
        r       minimal neighborhood size                                       -1 (no minimum)
        f       factor to increase the size of the neighborhood by              1.2
                if not enough neighbors were found
        k       number of neighbors to use (this verison uses exactly the       30
                number of neighbors specified. If found more, only the #
                nearest will be used)
        n       number of iterations                                            -1 (number of points)
        I       invert the order of the time series.                            0 (no inversion, 1 for inversion)
                Is supposed to help find spurious exponents.
    
        V:      verbosity level
        
            0: only panic messages
            1: add input/output messages 
            2: add information about the current state of the program
    
    Output: 
        
        The output consists of d*m+1 columns: 
            The first one shows the actual iteration.
            The next d*m ones the estimates of the Lyapunov exponents in decreasing order. 
            The last lines show the average forecast error(s) of the local linear model, the average neighborhood size used for fitting the model and the last one the estimated Kaplan-Yorke dimension.
            Output is written every 10 seconds (real time), approximately.
    """
    
    if (l == -1):
        l = len(data)
    
    if (n == -1):
        n = len(data)
    
    if (I ==0):
        if (r == -1):
            acf, msg = tiseanio('lyap_spec','-l',l,'-x',x,'-c',c,'-m',m,'-f',f,'-k',k,'-n',n,'-V',V,data=data)
        else:
            acf, msg = tiseanio('lyap_spec','-l',l,'-x',x,'-c',c,'-m',m,'-r',r,'-f',f,'-k',k,'-n',n,'-V',V,data=data)
    else:
        if (r == -1):
            acf, msg = tiseanio('lyap_spec','-l',l,'-x',x,'-c',c,'-m',m,'-f',f,'-k',k,'-n',n,'-I','-V',V,data=data)
        else:
            acf, msg = tiseanio('lyap_spec','-l',l,'-x',x,'-c',c,'-m',m,'-r',r,'-f',f,'-k',k,'-n',n,'-I','-V',V,data=data)
    
    return acf