#ifndef __dwi_tractography_connectomics_connectome_h__
#define __dwi_tractography_connectomics_connectome_h__

#include "math/matrix.h"


namespace MR
{

namespace DWI
{

namespace Tractography
{

namespace Connectomics
{


class NodePair
{

  public:

    NodePair();
    virtual ~NodePair();

    void setNodePair( const uint32_t firstNode,
                      const uint32_t secondNode );
    const uint32_t& getFirstNode() const;    
    const uint32_t& getSecondNode() const;

  protected:

    std::pair< uint32_t, uint32_t > _nodePair;

};


class Connectome
{

  public:

    Connectome( const uint32_t nodeCount );
    virtual ~Connectome();

    void update( const NodePair& nodePair );
    // functor for multi-thread
    bool operator() ( const NodePair& nodePair );

    void write( const std::string& path );

  protected:

    uint32_t _nodeCount;
    Math::Matrix< float > _matrix;

};


}

}

}

}


#endif
