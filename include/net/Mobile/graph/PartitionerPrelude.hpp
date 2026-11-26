#pragma once
// PartitionerPrelude.hpp — third-party partitioner headers that must be seen BEFORE TensorF's global
// `template<class T> class Graph` (KaMinPar's header uses `const class Graph&`, which would otherwise
// resolve to our template). Every header of this module includes it first; if you include
// DataStructures/Graph.hpp yourself, include net/Mobile headers before it.
#ifdef TENSORF_WITH_KAMINPAR
#include <kaminpar-shm/kaminpar.h>
#endif
#ifdef TENSORF_WITH_METIS
#include <metis.h>
#endif
#ifdef TENSORF_WITH_SCOTCH
#include <cstdio>
#include <scotch.h>
#endif
