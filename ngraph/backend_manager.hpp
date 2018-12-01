//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cstddef> // std::size_t, std::uintptr_t
#include <map>     // std::map
#include <mutex>   // std::mutex
#include "onnxifi.h"

#include "backend.hpp"
#include "ngraph/runtime/backend.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        /// \brief ONNXIFI backend manager
        class BackendManager
        {
        public:
            BackendManager(const BackendManager&) = delete;
            BackendManager& operator=(const BackendManager&) = delete;

            BackendManager(BackendManager&&) = delete;
            BackendManager& operator=(BackendManager&&) = delete;

            static void get_backend_ids(::onnxBackendID* backend_ids, std::size_t* count)
            {
                instance().get_registered_ids(backend_ids, count);
            }

            static void unregister(::onnxBackendID backend_id)
            {
                instance().unregister_backend(backend_id);
            }

            static const Backend& get(::onnxBackendID backend_id)
            {
                return instance().get_backend(backend_id);
            }

        private:
            mutable std::mutex m_mutex{};
            std::map<std::uintptr_t, Backend> m_registered_backends{};

            BackendManager();

            static BackendManager& instance()
            {
                static BackendManager backend_manager;
                return backend_manager;
            }

            void unregister_backend(std::uintptr_t id)
            {
                std::lock_guard<decltype(m_mutex)> lock{m_mutex};
                m_registered_backends.erase(id);
            }

            void unregister_backend(::onnxBackendID id)
            {
                return unregister_backend(reinterpret_cast<std::uintptr_t>(id));
            }

            void get_registered_ids(::onnxBackendID* backend_ids, std::size_t* count) const;

            const Backend& get_backend(std::uintptr_t id) const
            {
                std::lock_guard<decltype(m_mutex)> lock{m_mutex};
                return m_registered_backends.at(id);
            }

            const Backend& get_backend(::onnxBackendID id) const
            {
                return get_backend(reinterpret_cast<std::uintptr_t>(id));
            }
        };

    } // namespace onnxifi

} // namespace ngraph
