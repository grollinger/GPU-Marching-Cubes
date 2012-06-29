#ifndef GPU_MC_H_
#define GPU_MC_H_
#include <iostream>
#include <fstream>
#include <utility>
#include <string>
#include <math.h>
#include <boost/filesystem.hpp>
#include "WrapCL/OpenCL.hpp"
#include "VoxelSpace.h"
#include "Mesh.h"

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

namespace StereoFu{
	
	class GpuMC {
	private:
		size_t dim_exp;
		cl::Kernel constructHPLevelKernel;
		cl::Kernel classifyCubesKernel;
		cl::Kernel traverseHPKernel;
		std::vector<cl::Image3D> images;
		OpenCL ctx;
		cl::Program p;
		cl::CommandQueue q;

		cl::Program getProgram();
	public:
		GpuMC(OpenCL ctx, size_t dim_exp) : dim_exp(dim_exp), ctx(ctx),p(getProgram()), q(ctx.createQueue()) {

		}
		Mesh extract_isosurface(VoxelSpace space);
	}
};
#endif
