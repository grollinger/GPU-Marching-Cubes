#ifndef GPU_MC_H_
#define GPU_MC_H_
#include "StereoFuConfig.h"
#include <boost/filesystem.hpp>
#include "WrapCL/OpenCL.hpp"
#include "VoxelSpace.h"
#include "Mesh.h"

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

namespace StereoFu{
	
	class GpuMC {
	private:
		size_t dim_exp;
		OpenCL ctx;
		cl::Program p;
		cl::CommandQueue q;		
		cl::Kernel constructHPLevelUInt, constructHPLevelChar, constructHPLevelShort, traverseHP, classifyCubes;
		std::vector<cl::Image3D> images;
		

		cl::Program getProgram();

	public:
		GpuMC(OpenCL ctx, size_t dim_exp) : dim_exp(dim_exp), ctx(ctx),p(getProgram()), q(ctx.createQueue()),
			constructHPLevelUInt(p, "constructHPLevelUInt"),
			constructHPLevelChar(p, "constructHPLevelChar"),
			constructHPLevelShort(p, "constructHPLevelShort"),
			traverseHP(p, "traverseHP"),			
			classifyCubes(p, "classifyCubes")
		{

		}
		Mesh extract_isosurface(VoxelSpace space);
	};
}
#endif
