#include "gpu-mc.hpp"

namespace StereoFu{
	cl::Program GpuMC::getProgram(){
		using namespace std;
		boost::filesystem::path mc_source(GPU_MC_PATH);

		vector<boost::filesystem::path> sources;
		sources.push_back(mc_source);
		return ctx.buildProgram(sources);
	}
}


/*
void renderScene() {
    histoPyramidConstruction();

    // Read top of histoPyramid an use this size to allocate VBO below
	int sum[8] = {0,0,0,0,0,0,0,0};
    queue.enqueueReadImage(images[images.size()-1], CL_FALSE, origin, region, 0, 0, sum);

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	queue.finish();
	int totalSum = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7] ;
    
	if(totalSum == 0) {
		std::cout << "HistoPyramid result is 0" << std::endl;
        return;
	}
	
	// 128 MB
	//if(totalSum >= 1864135) // Need to split into several VBO's to support larger structures
	//	isolevel_up = true;

	// Create new VBO
	glGenBuffers(1, &VBO_ID);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_ID);
	glBufferData(GL_ARRAY_BUFFER, totalSum*18*sizeof(cl_float), NULL, GL_STATIC_DRAW);
	//std::cout << "VBO using: " << sum[0]*18*sizeof(cl_float) / (1024*1024) << " M bytes" << std::endl;
	glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Traverse the histoPyramid and fill VBO
    histoPyramidTraversal(totalSum);

    // Render VBO
    glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//glRotatef(270.0f, 1.0f, 0.0f, 0.0f);	
	drawFPSCounter(totalSum);

	glTranslatef(-camX, -camY, -camZ);

	glRotatef(xrot,1.0,0.0,0.0);
	glRotatef(yrot,0.0, 1.0, 0.0);

    

    glPushMatrix();
    glColor3f(1.0f, 1.0f, 1.0f);
    glScalef(scalingFactor.x, scalingFactor.y, scalingFactor.z);
    glTranslatef(translation.x, translation.y, translation.z);

    glRotatef(90.0f, 0.0f, 0.0f, 1.0f);
    // Normal Buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO_ID);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glVertexPointer(3, GL_FLOAT, 24, BUFFER_OFFSET(0));
	glNormalPointer(GL_FLOAT, 24, BUFFER_OFFSET(12));    

	queue.finish();
	//glWaitSync(traversalSync, 0, GL_TIMEOUT_IGNORED);
    glDrawArrays(GL_TRIANGLES, 0, totalSum*3);
	
    // Release buffer
    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glPopMatrix();
    glutSwapBuffers();
    glDeleteBuffers(1, &VBO_ID);
	
    angle += 0.1f;

}

int prepareDataset(uchar ** voxels, int sizeX, int sizeY, int sizeZ) {
    // If all equal and power of two exit
    if(sizeX == sizeY && sizeY == sizeZ && sizeX == pow(2, log2(sizeX)))
        return sizeX;

    // Find largest size and find closest power of two
    int largestSize = max(sizeX, max(sizeY, sizeZ));
    int size = 0;
    int i = 1;
    while(pow(2, i) < largestSize)
        i++;
    size = pow(2, i);

    // Make new voxel array of this size and fill it with zeros
    uchar * newVoxels = new uchar[size*size*size];
    for(int j = 0; j < size*size*size; j++) 
        newVoxels[j] = 0;

    // Fill the voxel array with previous data
    for(int x = 0; x < sizeX; x++) {
        for(int y = 0; y < sizeY; y++) {
            for(int z = 0; z <sizeZ; z++) {
                newVoxels[x + y*size + z*size*size] = voxels[0][x + y*sizeX + z*sizeX*sizeY];
            }
        }
    }
    delete[] voxels[0];
    voxels[0] = newVoxels;
    return size;
}

#include <sstream>

template <class T>
inline std::string to_string(const T& t) {
    std::stringstream ss;
    ss << t;
    return ss.str();
}

void setupOpenCL(uchar * voxels, int size) {
    SIZE = size; 
   try { 
        // Create a context that use a GPU and OpenGL interop.
		context = createCLGLContext(CL_DEVICE_TYPE_GPU, VENDOR_ANY);

        // Get a list of devices on this platform
		vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Create a command queue and use the first device
        queue = CommandQueue(context, devices[0]);

        // Read source file
        std::ifstream sourceFile("gpu-mc.cl");
        if(sourceFile.fail()) {
            std::cout << "Failed to open OpenCL source file" << std::endl;
            exit(-1);
        }
        std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));
        
        // Insert size
        int pos = sourceCode.find("**HP_SIZE**");
        sourceCode = sourceCode.replace(pos, 11, to_string(SIZE));
        Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

        // Make program of the source code in the context
        program = Program(context, source);
    
        // Build program for these specific devices
        try{
            program.build(devices);
        } catch(Error error) {
            if(error.err() == CL_BUILD_PROGRAM_FAILURE) {
                std::cout << "Build log:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
            }   
            throw error;
        } 

        // Create images for the HistogramPyramid
        int bufferSize = SIZE;
		// Make the two first buffers use INT8
		images.push_back(Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), bufferSize, bufferSize, bufferSize));
		bufferSize /= 2;
		images.push_back(Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_UNSIGNED_INT8), bufferSize, bufferSize, bufferSize));
		bufferSize /= 2;
		// And the third, fourth and fifth INT16
		images.push_back(Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_UNSIGNED_INT16), bufferSize, bufferSize, bufferSize));
		bufferSize /= 2;
		images.push_back(Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_UNSIGNED_INT16), bufferSize, bufferSize, bufferSize));
		bufferSize /= 2;
		images.push_back(Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_UNSIGNED_INT16), bufferSize, bufferSize, bufferSize));
		bufferSize /= 2;
        // The rest will use INT32
        for(int i = 5; i < (log2((float)SIZE)); i ++) {
			if(bufferSize == 1)
				bufferSize = 2; // Image cant be 1x1x1
			images.push_back(Image3D(context, CL_MEM_READ_WRITE, ImageFormat(CL_R, CL_UNSIGNED_INT32), bufferSize, bufferSize, bufferSize));
            bufferSize /= 2;
        }

        // Transfer dataset to device
		rawData = Image3D(
                context, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                ImageFormat(CL_R, CL_UNSIGNED_INT8), 
                SIZE, SIZE, SIZE,
                0, 0, voxels
        );
        delete[] voxels;

		// Make kernels
		constructHPLevelKernel = Kernel(program, "constructHPLevel");
		classifyCubesKernel = Kernel(program, "classifyCubes");
		traverseHPKernel = Kernel(program, "traverseHP");

    } catch(Error error) {
       std::cout << error.what() << "(" << error.err() << ")" << std::endl;
       std::cout << getCLErrorString(error.err()) << std::endl;
    }
}


void histoPyramidConstruction() {

        updateScalarField();

        // Run base to first level
		constructHPLevelKernel.setArg(0, images[0]);
		constructHPLevelKernel.setArg(1, images[1]);

        queue.enqueueNDRangeKernel(
			constructHPLevelKernel, 
			NullRange, 
			NDRange(SIZE/2, SIZE/2, SIZE/2), 
			NullRange
		);

        int previous = SIZE / 2;
        // Run level 2 to top level
        for(int i = 1; i < log2((float)SIZE)-1; i++) {
			constructHPLevelKernel.setArg(0, images[i]);
			constructHPLevelKernel.setArg(1, images[i+1]);
			previous /= 2;
            queue.enqueueNDRangeKernel(
				constructHPLevelKernel, 
				NullRange, 
				NDRange(previous, previous, previous), 
                NullRange
			);
        }
}

void updateScalarField() {
    classifyCubesKernel.setArg(0, images[0]);
    classifyCubesKernel.setArg(1, rawData);
	classifyCubesKernel.setArg(2, isolevel);
    queue.enqueueNDRangeKernel(
            classifyCubesKernel, 
            NullRange, 
            NDRange(SIZE, SIZE, SIZE),
            NullRange
    );
}

BufferGL VBOBuffer;
void histoPyramidTraversal(int sum) {
    // Make OpenCL buffer from OpenGL buffer
	unsigned int i = 0;
	for(i = 0; i < images.size(); i++) {
		traverseHPKernel.setArg(i, images[i]);
	}
	
	VBOBuffer = BufferGL(context, CL_MEM_WRITE_ONLY, VBO_ID);
    traverseHPKernel.setArg(i, VBOBuffer);
	traverseHPKernel.setArg(i+1, isolevel);
	traverseHPKernel.setArg(i+2, sum);
	//cl_event syncEvent = clCreateEventFromGLsyncKHR((cl_context)context(), (cl_GLsync)glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0), 0);
	//glFinish();
	vector<Memory> v;
	v.push_back(VBOBuffer);
	//vector<Event> events;
	//Event e;
	//events.push_back(Event(syncEvent));
    queue.enqueueAcquireGLObjects(&v);

	// Increase the global_work_size so that it is divideable by 64
	int global_work_size = sum + 64 - (sum - 64*(sum / 64));
    // Run a NDRange kernel over this buffer which traverses back to the base level
    queue.enqueueNDRangeKernel(traverseHPKernel, NullRange, NDRange(global_work_size), NDRange(64));

	Event traversalEvent;	
    queue.enqueueReleaseGLObjects(&v, 0, &traversalEvent);
//	traversalSync = glCreateSyncFromCLeventARB((cl_context)context(), (cl_event)traversalEvent(), 0); // Need the GL_ARB_cl_event extension
    queue.flush();
}*/
