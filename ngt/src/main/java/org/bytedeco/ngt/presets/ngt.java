/*
 * Copyright (C) 2017-2020 Jeremy Apthorp, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.ngt.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.javacpp.tools.*;

@Properties(
    inherit = javacpp.class,
    target = "org.bytedeco.ngt",
    global = "org.bytedeco.ngt.global.ngt",
    value = {
        @Platform(
            value = {"linux-x86_64"},
	    define = {
		    "SHARED_PTR_NAMESPACE std",
	   	    "UNIQUE_PTR_NAMESPACE std",
		    "NGT_SHARED_MEMORY_ALLOCATOR ON",
		    "NGT_LARGE_DATASET ON"},
            include = {
		// "ObjectRepository.h",
           	// "Graph.h",

		// "NGT/SharedMemoryAllocator.h",
		// "NGT/defines.h",
  	        "NGT/Common.h",
		// "NGT/MmapManager.h",
		// "NGT/ObjectSpaceRepository.h",
		// "NGT/ObjectSpace.h",
		// "NGT/PrimitiveComparator.h",
		// "NGT/Node.h",
           	// "NGT/Tree.h",
           	// "NGT/Thread.h",
		// "NGT/Capi.h",
                "NGT/Index.h"
            }
	    ,
            compiler = "cpp11",
            link = "ngt",
            preload = "libngt"
        )
    }
)
public class ngt implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "ngt"); }

    public void map(InfoMap infoMap) {
        infoMap
            // .put(new Info("NGT::ObjectRepository::append(float *, size_t)").javaNames("appendFloat"))
            // .put(new Info("NGT::ObjectRepository::allocateObject(float *, size_t)").javaNames("allocateFloatObject"))
            // .put(new Info("NGT::ObjectRepository::allocateObject(const std::vector<float>&)").javaNames("allocateFloatVectorObject"))
            // .put(new Info("NGT::ObjectRepository::allocatePersistentObject(float *, size_t)").javaNames("allocateFloatPersistentObject"))
            // .put(new Info("NGT::ObjectRepository::allocatePersistentObject(const std::vector<float>&)").javaNames("allocateFloatVectorPersistentObject"))

	    .put(new Info("defined(NGT_SHARED_MEMORY_ALLOCATOR)").define(true))

	    .put(new Info("NGT::Index::append(const std::vector<Float>&)"))
	    .put(new Info("NGT::Index::insert(const std::vector<Float>&)"))
	    .put(new Info("NGT::Index::remove").skip())
	    .put(new Info("NGT::Index::save").skip())
	    .put(new Info("NGT::Index::Index").skip())

      	    .put(new Info("SharedMemoryAllocator::operator").skip())

            .put(new Info("NGT::DVPTree").skip())
            .put(new Info("NGT::InternalNode").skip())
            .put(new Info("NGT::QueryContainer").skip())
            .put(new Info("NGT::QueryContainer").skip())
            .put(new Info("NGT::Timer").skip())
            .put(new Info("NGT::Node").skip())
            .put(new Info("NGT::SearchContainer"))
            .put(new Info("NGT::SharedMemoryAllocator").skip())
            .put(new Info("NGT::GraphIndex").skip())
            .put(new Info("NGT::LeafNode").skip())
            .put(new Info("NGT::StdOstreamRedirector").skip())
	    .put(new Info("operator new").skip())
	    .put(new Info("operator new[]").skip())

            .put(new Info("SharedMemoryAllocator::GetMemorySizeType").skip())
            .put(new Info("NGT::ObjectID").cast().valueTypes("int"))
			    // pointerTypes("ObjectID").define())
            // .put(new Info("NGT::ObjectDistance").pointerTypes("ngtObjectDist"))
            .put(new Info("std::type_info").cast().pointerTypes("Pointer"))
      	    .put(new Info("std::pair<float,double>").pointerTypes("pairFloatDouble").define())
	    .put(new Info("std::pair<NGT::Object*,size_t>").pointerTypes("pairNGTObjectSize").define())
	    .put(new Info("std::vector<NGT::ObjectDistance>").pointerTypes("vectorObjectDistance").define())
	    .put(new Info("std::map<std::string,std::string>").pointerTypes("mapStringString").define())
	    .put(new Info("std::set<std::string>").pointerTypes("setString").define())
	    .put(new Info("std::vector<float>").pointerTypes("vecfloat").define())
	    // .put(new Info("NGT::PersistentRepository<NGT::InternalNode>").pointerTypes("prIn").define())
	    // .put(new Info("NGT::PersistentRepository<NGT::LeafNode>").pointerTypes("prLn").define())

	    // .put(new Info("std::priority_queue<NGT::ObjectDistance,std::vector<NGT::ObjectDistance>,std::less<NGT::ObjectDistance> >").pointerTypes("priorityQueueODvOD").define())
	    .put(new Info("NGT::ObjectDistances::moveFrom").skip())
	    .put(new Info("NGT::ObjectSpace::linearSearch").skip())
	    .put(new Info("NGT::ObjectSpace").skip())
	    .put(new Info("NGT::PersistentObject").skip())
	    .put(new Info("NGT::PersistentObjectDistances").skip())
	    .put(new Info("NGT::ResultPriorityQueue").skip())
	    .put(new Info("NGT::Vector<Float>").skip())

            .put(new Info("NGT::Object"))
            .put(new Info("NGT::Common").skip())
            // .put(new Info("NGT::ObjectID"))
            .put(new Info("NGT::Index").javaNames("ngtIndex"))
            .put(new Info("NGT::ObjectSpace"))
            .put(new Info("NGT::GraphIndex::NeighborhoodGraph").skip())
      	    .put(new Info("NGT::Index::Property").skip())
      	    .put(new Info("NGT::Index::Property::DistanceType").skip())
      	    .put(new Info("NGT::Index::Property::ObjectType").skip())
            .put(new Info("NGT::ObjectSpace::ObjectType::Float").javaNames("ObjectType_Float"))
            // .put(new Info("NGT::ObjectSpace").skip())
            .put(new Info("NGT::Index::objectType").skip())

            .put(new Info("NGT::SearchQuery").skip())
            // .put(new Info("NGT::SearchQuery(const std::vector<Float> &)").javaNames("searchQueryFloat"))
	    .put(new Info("NGT::GraphOptimizer"))
            ;
    }
}
