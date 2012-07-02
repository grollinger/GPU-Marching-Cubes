#ifndef STEREOFU_EXTRACTION_H_
#define STEREOFU_EXTRACTION_H_
#include "StereoFuConfig.hpp"
#include <boost/filesystem.hpp>
#include "OpenCL.hpp"
#include "VoxelSpace.hpp"
#include "Mesh.hpp"

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

namespace StereoFu{
	
	class Extraction {
	private:
		size_t dim_exp;
		OpenCL ctx;
		cl::Program p;
		cl::CommandQueue q;		
		cl::Kernel constructHPLevelUInt, constructHPLevelChar, constructHPLevelShort, traverseHP, classifyCubes;
		
		

		cl::Program getProgram();


	public:
		Extraction(OpenCL ctx, size_t dim_exp) : dim_exp(dim_exp), ctx(ctx),p(getProgram()), q(ctx.createQueue()),
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
