package halo

import "strings"

func GetHaloKernels(cfg Config) string {
	return strings.Replace(haloKernelSource, "DTYPE", cfg.DataType, -1)
}

var haloKernelSource = `
// Simple gather kernel - works on all backends
@kernel void simpleGatherFaces(const int nFaces,
                               const int Np,
                               const int Nfp,
                               @restrict const int *elements,
                               @restrict const int *faces,
                               @restrict const int *fmask,
                               @restrict const DTYPE *Q,
                               @restrict DTYPE *sendBuffer) {
    for (int i = 0; i < nFaces; ++i; @outer) {
        for (int fp = 0; fp < Nfp; ++fp; @inner) {
            const int elem = elements[i];
            const int face = faces[i];
            const int volPoint = fmask[face * Nfp + fp];
            
            sendBuffer[i * Nfp + fp] = Q[elem * Np + volPoint];
        }
    }
}

// Simple scatter kernel - works on all backends
// Updated to write to face-contiguous buffer layout
@kernel void simpleScatterFaces(const int nFaces,
                                const int Np,
                                const int Nfp,
                                const int Nface,
                                @restrict const int *elements,
                                @restrict const int *faces,
                                @restrict const int *fmask,
                                @restrict const DTYPE *recvBuffer,
                                @restrict DTYPE *Qghost) {
    for (int i = 0; i < nFaces; ++i; @outer) {
        for (int fp = 0; fp < Nfp; ++fp; @inner) {
            const int elem = elements[i];
            const int face = faces[i];
            
            // Write to face-contiguous layout: [element][face][face_point]
            Qghost[elem * Nface * Nfp + face * Nfp + fp] = recvBuffer[i * Nfp + fp];
        }
    }
}

// Direct local exchange - works on all backends
// Updated to read from volume buffer and write to face-contiguous buffer
@kernel void simpleLocalExchange(const int nPairs,
                                 const int Np,
                                 const int Nfp,
                                 const int Nface,
                                 @restrict const int *sendElems,
                                 @restrict const int *sendFaces,
                                 @restrict const int *recvElems,
                                 @restrict const int *recvFaces,
                                 @restrict const int *fmask,
                                 @restrict const DTYPE *Q,
                                 @restrict DTYPE *Qghost) {
    for (int pair = 0; pair < nPairs; ++pair; @outer) {
        for (int fp = 0; fp < Nfp; ++fp; @inner) {
            const int srcElem = sendElems[pair];
            const int srcFace = sendFaces[pair];
            const int dstElem = recvElems[pair];
            const int dstFace = recvFaces[pair];
            
            // Read from volume point in Q
            const int srcPoint = fmask[srcFace * Nfp + fp];
            
            // Write to face-contiguous layout in Qghost
            Qghost[dstElem * Nface * Nfp + dstFace * Nfp + fp] = Q[srcElem * Np + srcPoint];
        }
    }
}
`
