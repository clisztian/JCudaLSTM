package ch.imetrica.recurrentnn.loss;
import static jcuda.driver.JCudaDriver.*;

import java.io.*;
import java.util.Random;

import jcuda.*;
import jcuda.driver.*;

/**
 * Example of a reduction. It is based on the NVIDIA 'reduction' sample, 
 * and uses an adopted version of one of the kernels presented in 
 * this sample. 
 */
public class JCudaReduction
{
    /**
     * The CUDA context created by this sample
     */
    private static CUcontext context;
    
    /**
     * The module which is loaded in form of a PTX file
     */
    private static CUmodule module;
    
    /**
     * The actual kernel function from the module
     */
    private static CUfunction function;
    
    /**
     * Temporary memory for the device output
     */
    private static CUdeviceptr deviceBuffer;
    
    /**
     * Entry point of this sample
     *
     * @param args Not used
     */
    public static void main(String args[])
    {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        init();
        boolean passed = true;
        for (int n = 100000; n <= 25600000; n *= 2)
        {
            double hostInput[] = createRandomArray(n);

            long time0 = 0;
            long time1 = 0;

            // Copy the input data to the device
            time0 = System.nanoTime();
            CUdeviceptr deviceInput = new CUdeviceptr();
            cuMemAlloc(deviceInput, hostInput.length * Sizeof.DOUBLE);
            cuMemcpyHtoD(deviceInput, Pointer.to(hostInput), 
                hostInput.length * Sizeof.DOUBLE);
            time1 = System.nanoTime();
            long durationCopy = time1 - time0;

            // Execute the reduction with CUDA
            time0 = System.nanoTime();
            double resultJCuda = reduce(deviceInput, hostInput.length);
            time1 = System.nanoTime();
            long durationComp = time1 - time0;

            cuMemFree(deviceInput);

            // Execute the reduction with Java
            time0 = System.nanoTime();
            double resultJava = reduceHost(hostInput);
            time1 = System.nanoTime();
            long durationJava = time1 - time0;

            System.out.println("Reduction of " + n + " elements");
            System.out.printf(
                "  JCuda: %5.3fms, result: %f " +
                "(copy: %5.3fms, comp: %5.3fms)\n",
                (durationCopy + durationComp) / 1e6, resultJCuda, 
                durationCopy / 1e6, durationComp / 1e6);
            System.out.printf(
                "  Java : %5.3fms, result: %f\n", 
                durationJava / 1e6, resultJava);
            
            passed &= 
                Math.abs(resultJCuda - resultJava) < resultJava * 1e-5;
            
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));

        shutdown();
    }    
    
    
    
    
    /**
     * Implementation of a Kahan summation reduction in plain Java
     * 
     * @param input The input 
     * @return The reduction result
     */
    static double reduceHost(double data[])
    {
        double sum = data[0];
        double c = 0.0;              
        for (int i = 1; i < data.length; i++)
        {
            double y = data[i] - c;  
            double t = sum + y;      
            c = (t - sum) - y;  
            sum = t;            
        }
        return sum;
    }
    
    
    /**
     * Initialize the driver API and create a context for the first
     * device, and then call {@link #prepare()}
     */
    private static void init()
    {
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
        prepare();
    }
    
    /**
     * Prepare everything for calling the reduction kernel function.
     * This method assumes that a context already has been created
     * and is current!
     */
    public static void prepare()
    {
        // Prepare the ptx file.
        String ptxFileName = null;
        try
        {
            ptxFileName = preparePtxFile("cuda/exp_reduction.cu");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Could not prepare PTX file", e);
        }
        
        // Load the module from the PTX file
        module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "reduce" function.
        function = new CUfunction();
        cuModuleGetFunction(function, module, "reduce_exp");
        
        // Allocate a chunk of temporary memory (must be at least
        // numberOfBlocks * Sizeof.DOUBLE)
        deviceBuffer = new CUdeviceptr();
        cuMemAlloc(deviceBuffer, 1024 * Sizeof.DOUBLE);
        
    }
    
    /**
     * Release all resources allocated by this class
     */
    public static void shutdown()
    {
        cuModuleUnload(module);
        cuMemFree(deviceBuffer);
        if (context != null)
        {
            cuCtxDestroy(context);
        }
    }
    
    /**
     * Perform a reduction on the given input, with a default number
     * of threads and blocks, and return the result. <br />
     * <br />
     * This method assumes that either {@link #init()} or 
     * {@link #prepare()} have already been called.
     * 
     * @param hostInput The input to reduce
     * @return The reduction result
     */
    public static double reduce(double hostInput[])
    {
        return reduce(hostInput, 128, 64);
    }
    
    /**
     * Perform a reduction on the given input, with the given number
     * of threads and blocks, and return the result. <br /> 
     * <br />
     * This method assumes that either {@link #init()} or 
     * {@link #prepare()} have already been called.
     * 
     * @param hostInput The input to reduce
     * @param maxThreads The maximum number of threads per block
     * @param maxBlocks The maximum number of blocks per grid
     * @return The reduction result
     */
    public static double reduce(
        double hostInput[], int maxThreads, int maxBlocks)
    {
        // Allocate and fill the device memory
        CUdeviceptr deviceInput = new CUdeviceptr();
        cuMemAlloc(deviceInput, hostInput.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceInput, Pointer.to(hostInput), 
            hostInput.length * Sizeof.DOUBLE);

        // Call reduction on the device memory
        double result = 
            reduce(deviceInput, hostInput.length, maxThreads, maxBlocks);

        // Clean up and return the result
        cuMemFree(deviceInput);
        return result;
    }

    
    /**
     * Performs a reduction on the given device memory with the given
     * number of elements.
     * 
     * @param deviceInput The device input memory
     * @param numElements The number of elements to reduce
     * @return The reduction result
     */
    public static double reduce(
        Pointer deviceInput, int numElements)
    {
        return reduce(deviceInput, numElements, 128, 64);
    }
    
    
    /**
     * Performs a reduction on the given device memory with the given
     * number of elements and the specified limits for threads and
     * blocks.
     * 
     * @param deviceInput The device input memory
     * @param numElements The number of elements to reduce
     * @param maxThreads The maximum number of threads
     * @param maxBlocks The maximum number of blocks
     * @return The reduction result
     */
    public static double reduce(
        Pointer deviceInput, int numElements, 
        int maxThreads, int maxBlocks)
    {
        // Determine the number of threads and blocks 
        // (as done in the NVIDIA sample)
        int numBlocks = getNumBlocks(numElements, maxBlocks, maxThreads);
        int numThreads = getNumThreads(numElements, maxBlocks, maxThreads);
        
        // Call the main reduction method
        double result = reduce(numElements, numThreads, numBlocks, 
            maxThreads, maxBlocks, deviceInput);
        return result;
    }
    

    
    /**
     * Performs a reduction on the given device memory.
     * 
     * @param n The number of elements for the reduction
     * @param numThreads The number of threads
     * @param numBlocks The number of blocks
     * @param maxThreads The maximum number of threads
     * @param maxBlocks The maximum number of blocks
     * @param deviceInput The input memory
     * @return The reduction result
     */
    private static double reduce(
        int  n, int  numThreads, int  numBlocks,
        int  maxThreads, int  maxBlocks, Pointer deviceInput)
    {
        // Perform a "tree like" reduction as in the NVIDIA sample
        reduce(n, numThreads, numBlocks, deviceInput, deviceBuffer);
        int s=numBlocks;
        while(s > 1) 
        {
            int threads = getNumThreads(s, maxBlocks, maxThreads);
            int blocks = getNumBlocks(s, maxBlocks, maxThreads);

            reduce(s, threads, blocks, deviceBuffer, deviceBuffer);
            s = (s + (threads*2-1)) / (threads*2);
        }
        
        double result[] = {0.0};
        cuMemcpyDtoH(Pointer.to(result), deviceBuffer, Sizeof.DOUBLE);     
        return result[0];
    }
    
    
    /**
     * Perform a reduction of the specified number of elements in the given 
     * device input memory, using the given number of threads and blocks, 
     * and write the results into the given output memory. 
     * 
     * @param size The size (number of elements) 
     * @param threads The number of threads
     * @param blocks The number of blocks
     * @param deviceInput The device input memory
     * @param deviceOutput The device output memory. Its size must at least 
     * be numBlocks*Sizeof.DOUBLE
     */
    private static void reduce(int size, int threads, int blocks, 
        Pointer deviceInput, Pointer deviceOutput)
    {
        //System.out.println("Reduce "+size+" elements with "+
        //    threads+" threads in "+blocks+" blocks");
        
        // Compute the shared memory size (as done in 
        // the NIVIDA sample)
        int sharedMemSize = threads * Sizeof.DOUBLE;
        if (threads <= 32) 
        {
            sharedMemSize *= 2;
        }
        
        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
            Pointer.to(deviceInput),
            Pointer.to(deviceOutput),
            Pointer.to(new int[]{size})
        );

        // Call the kernel function.
        cuLaunchKernel(function,
            blocks,  1, 1,         // Grid dimension
            threads, 1, 1,         // Block dimension
            sharedMemSize, null,   // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
    }
    
    
    /**
     * Compute the number of blocks that should be used for the
     * given input size and limits
     * 
     * @param n The input size
     * @param maxBlocks The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of blocks
     */
    private static int getNumBlocks(int n, int maxBlocks, int maxThreads)
    {
        int blocks = 0;
        int threads = getNumThreads(n, maxBlocks, maxThreads);
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
        blocks = Math.min(maxBlocks, blocks);
        return blocks;
    }

    /**
     * Compute the number of threads that should be used for the
     * given input size and limits
     * 
     * @param n The input size
     * @param maxBlocks The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of threads
     */
    private static int getNumThreads(int n, int maxBlocks, int maxThreads)
    {
        int threads = 0;
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        return threads;
    }
    
    /**
     * Returns the power of 2 that is equal to or greater than x
     * 
     * @param x The input
     * @return The next power of 2
     */
    private static int nextPow2(int x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    
    /**
     * Create an array of the given size, with random data
     * 
     * @param size The array size
     * @return The array
     */
    private static double[] createRandomArray(int size)
    {
        Random random = new Random(0);
        double array[] = new double[size];
        for(int i = 0; i < size; i++)
        {
            array[i] = random.nextDouble();
        }
        return array;
    }
    

    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is
     * compiled from the given file using NVCC. The name of the
     * PTX file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private static String preparePtxFile(String cuFileName) throws IOException
    {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists())
        {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            "nvcc " + modelString + " -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;

        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
            new String(toByteArray(process.getErrorStream()));
        String outputMessage =
            new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                "Could not create .ptx file: "+errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream)
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

    static double softmaxHost(double data[])
    {
        double sum = 0;                   
        double max = 0;
        for (int i = 0; i < data.length; i++)
        {
        	sum += Math.exp(data[i] - max);              
        }
        return sum;
    }  
}