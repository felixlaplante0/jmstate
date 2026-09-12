#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct Bucket {
    std::vector<int64_t> idxs;
    std::vector<double> t0s;
    std::vector<double> t1s;
    std::vector<uint8_t> obs;
};

template <typename T>
py::array_t<T> to_array(const std::vector<T> &values) {
    py::array_t<T> out(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(T));
    }
    return out;
}

py::array_t<bool> to_mask(const std::vector<uint8_t> &values) {
    py::array_t<bool> out(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(out.mutable_data(), values.data(), values.size());
    }
    return out;
}

py::array to_time_array(const std::vector<double> &values, bool float64) {
    if (float64) {
        py::array_t<double> out(static_cast<py::ssize_t>(values.size()));
        if (!values.empty()) {
            std::memcpy(out.mutable_data(), values.data(),
                        values.size() * sizeof(double));
        }
        return out;
    }
    py::array_t<float> out(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        float *dst = out.mutable_data();
        for (size_t i = 0; i < values.size(); ++i) {
            dst[i] = static_cast<float>(values[i]);
        }
    }
    return out;
}

struct PyObjHash {
    size_t operator()(const py::object &obj) const {
        return static_cast<size_t>(py::hash(obj));
    }
};

struct PyObjEq {
    bool operator()(const py::object &lhs, const py::object &rhs) const {
        return lhs.equal(rhs);
    }
};

using AltMap =
    std::unordered_map<py::object, std::vector<py::ssize_t>, PyObjHash, PyObjEq>;

py::ssize_t find_or_create(py::dict &registry, std::vector<Bucket> &buckets,
                           std::vector<py::object> &keys, const py::tuple &key) {
    PyObject *existing = PyDict_GetItemWithError(registry.ptr(), key.ptr());
    if (existing != nullptr) {
        return py::cast<py::ssize_t>(py::reinterpret_borrow<py::object>(existing));
    }
    if (PyErr_Occurred()) {
        throw py::error_already_set();
    }
    const py::ssize_t index = static_cast<py::ssize_t>(buckets.size());
    registry[key] = index;
    buckets.emplace_back();
    keys.emplace_back(py::reinterpret_borrow<py::object>(key));
    return index;
}

py::dict _build_buckets(const py::list &trajectories, bool float64) {
    py::dict registry;
    std::vector<Bucket> buckets;
    std::vector<py::object> keys;

    const py::ssize_t n = py::len(trajectories);
    for (py::ssize_t i = 0; i < n; ++i) {
        const py::list trajectory = trajectories[i];
        const py::ssize_t m = py::len(trajectory);
        if (m == 0) {
            throw py::value_error("Trajectories must not be empty");
        }
        py::tuple p0 = py::cast<py::tuple>(trajectory[0]);
        for (py::ssize_t j = 1; j < m; ++j) {
            const py::tuple p1 = py::cast<py::tuple>(trajectory[j]);
            const double t0 = p0[0].cast<double>();
            const double t1 = p1[0].cast<double>();
            const py::tuple key = py::make_tuple(p0[1], p1[1]);
            const py::ssize_t index = find_or_create(registry, buckets, keys, key);
            Bucket &bucket = buckets[static_cast<size_t>(index)];
            bucket.idxs.push_back(static_cast<int64_t>(i));
            bucket.t0s.push_back(t0);
            bucket.t1s.push_back(t1);
            p0 = p1;
        }
    }

    py::dict out;
    for (size_t k = 0; k < buckets.size(); ++k) {
        const Bucket &bucket = buckets[k];
        out[keys[k]] = py::make_tuple(to_array(bucket.idxs),
                                      to_time_array(bucket.t0s, float64),
                                      to_time_array(bucket.t1s, float64));
    }
    return out;
}

void build_alt_map(const py::list &link_keys, AltMap &alt_map,
                   std::vector<py::object> &dest_states) {
    dest_states.reserve(static_cast<size_t>(py::len(link_keys)));
    for (const py::handle key_h : link_keys) {
        const py::tuple key = py::cast<py::tuple>(key_h);
        dest_states.emplace_back(py::reinterpret_borrow<py::object>(key[1]));
        py::object src = py::reinterpret_borrow<py::object>(key[0]);
        alt_map[std::move(src)].push_back(
            static_cast<py::ssize_t>(dest_states.size() - 1));
    }
}

py::dict _build_quad_buckets(const py::list &trajectories, const py::list &link_keys,
                             const py::list &censoring, bool float64) {
    AltMap alt_map;
    std::vector<py::object> dest_states;
    build_alt_map(link_keys, alt_map, dest_states);

    const size_t n_links = static_cast<size_t>(py::len(link_keys));
    std::vector<Bucket> buckets(n_links);
    std::vector<uint8_t> used(n_links, 0);
    std::vector<size_t> order;
    order.reserve(n_links);

    const py::ssize_t n = py::len(trajectories);
    for (py::ssize_t i = 0; i < n; ++i) {
        const py::list trajectory = trajectories[i];
        const py::ssize_t m = py::len(trajectory);
        if (m == 0) {
            throw py::value_error("Trajectories must not be empty");
        }
        py::tuple p0 = py::cast<py::tuple>(trajectory[0]);
        for (py::ssize_t j = 1; j < m; ++j) {
            const py::tuple p1 = py::cast<py::tuple>(trajectory[j]);
            const double t0 = p0[0].cast<double>();
            const double t1 = p1[0].cast<double>();
            const py::object s0 = py::reinterpret_borrow<py::object>(p0[1]);
            const py::object s1 = py::reinterpret_borrow<py::object>(p1[1]);
            const auto it = alt_map.find(s0);
            if (it != alt_map.end()) {
                for (const py::ssize_t k : it->second) {
                    const auto idx = static_cast<size_t>(k);
                    if (!used[idx]) {
                        used[idx] = 1;
                        order.push_back(idx);
                    }
                    Bucket &bucket = buckets[idx];
                    bucket.idxs.push_back(static_cast<int64_t>(i));
                    bucket.t0s.push_back(t0);
                    bucket.t1s.push_back(t1);
                    bucket.obs.push_back(dest_states[idx].equal(s1) ? 1 : 0);
                }
            }
            p0 = p1;
        }

        const double last_t = p0[0].cast<double>();
        const double c_i = censoring[i].cast<double>();
        if (last_t >= c_i) {
            continue;
        }
        const py::object last_s = py::reinterpret_borrow<py::object>(p0[1]);
        const auto it = alt_map.find(last_s);
        if (it == alt_map.end()) {
            continue;
        }
        for (const py::ssize_t k : it->second) {
            const auto idx = static_cast<size_t>(k);
            if (!used[idx]) {
                used[idx] = 1;
                order.push_back(idx);
            }
            Bucket &bucket = buckets[idx];
            bucket.idxs.push_back(static_cast<int64_t>(i));
            bucket.t0s.push_back(last_t);
            bucket.t1s.push_back(c_i);
            bucket.obs.push_back(0);
        }
    }

    py::dict out;
    for (const size_t k : order) {
        const Bucket &bucket = buckets[k];
        py::object key =
            py::reinterpret_borrow<py::object>(link_keys[static_cast<py::ssize_t>(k)]);
        out[key] = py::make_tuple(to_array(bucket.idxs),
                                  to_time_array(bucket.t0s, float64),
                                  to_time_array(bucket.t1s, float64),
                                  to_mask(bucket.obs));
    }
    return out;
}

py::dict _build_remaining_buckets(const py::list &trajectories,
                                  const py::list &link_keys,
                                  const py::list &censoring, bool float64) {
    AltMap alt_map;
    std::vector<py::object> dest_states;
    build_alt_map(link_keys, alt_map, dest_states);

    const size_t n_links = static_cast<size_t>(py::len(link_keys));
    std::vector<Bucket> buckets(n_links);
    std::vector<uint8_t> used(n_links, 0);
    std::vector<size_t> order;
    order.reserve(n_links);

    const py::ssize_t n = py::len(trajectories);
    for (py::ssize_t i = 0; i < n; ++i) {
        const py::list trajectory = trajectories[i];
        const py::ssize_t m = py::len(trajectory);
        if (m == 0) {
            throw py::value_error("Trajectories must not be empty");
        }
        const py::tuple last = py::cast<py::tuple>(trajectory[m - 1]);
        const double last_t = last[0].cast<double>();
        const double c_i = censoring[i].cast<double>();
        if (last_t >= c_i) {
            continue;
        }
        const py::object last_s = py::reinterpret_borrow<py::object>(last[1]);
        const auto it = alt_map.find(last_s);
        if (it == alt_map.end()) {
            continue;
        }
        for (const py::ssize_t k : it->second) {
            const auto idx = static_cast<size_t>(k);
            if (!used[idx]) {
                used[idx] = 1;
                order.push_back(idx);
            }
            Bucket &bucket = buckets[idx];
            bucket.idxs.push_back(static_cast<int64_t>(i));
            bucket.t0s.push_back(last_t);
        }
    }

    py::dict out;
    for (const size_t k : order) {
        const Bucket &bucket = buckets[k];
        py::object key =
            py::reinterpret_borrow<py::object>(link_keys[static_cast<py::ssize_t>(k)]);
        out[key] = py::make_tuple(to_array(bucket.idxs),
                                  to_time_array(bucket.t0s, float64));
    }
    return out;
}

PYBIND11_MODULE(_surv_ext, m, py::mod_gil_not_used()) {
    m.doc() = "C++ survival bucket construction for jmstate.";
    m.def("_build_buckets", &_build_buckets, py::arg("trajectories"),
          py::arg("float64"),
          "Group observed segments by (from_state, to_state).");
    m.def("_build_quad_buckets", &_build_quad_buckets, py::arg("trajectories"),
          py::arg("link_keys"), py::arg("censoring"), py::arg("float64"),
          "Build vectorizable buckets with competing transitions and tails.");
    m.def("_build_remaining_buckets", &_build_remaining_buckets,
          py::arg("trajectories"), py::arg("link_keys"), py::arg("censoring"),
          py::arg("float64"),
          "Build censored-tail buckets.");
}
