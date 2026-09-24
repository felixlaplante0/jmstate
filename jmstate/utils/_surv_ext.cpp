#include <cstddef>
#include <cstdint>
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

template <typename Out, typename In>
py::array_t<Out> to_array(const std::vector<In> &values) {
    py::array_t<Out> out(static_cast<py::ssize_t>(values.size()));
    Out *dst = out.mutable_data();
    for (size_t i = 0; i < values.size(); ++i) {
        dst[i] = static_cast<Out>(values[i]);
    }
    return out;
}

py::array to_time_array(const std::vector<double> &values, bool float64) {
    if (float64) {
        return to_array<double>(values);
    }
    return to_array<float>(values);
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

py::list checked_trajectory(const py::list &trajectories, py::ssize_t i) {
    py::list trajectory = trajectories[i];
    if (py::len(trajectory) == 0) {
        throw py::value_error("Trajectories must not be empty");
    }
    return trajectory;
}

py::dict _build_buckets(const py::list &trajectories, bool float64) {
    py::dict registry;
    std::vector<Bucket> buckets;
    std::vector<py::object> keys;

    const py::ssize_t n = py::len(trajectories);
    for (py::ssize_t i = 0; i < n; ++i) {
        const py::list trajectory = checked_trajectory(trajectories, i);
        const py::ssize_t m = py::len(trajectory);
        py::tuple p0 = py::cast<py::tuple>(trajectory[0]);
        for (py::ssize_t j = 1; j < m; ++j) {
            const py::tuple p1 = py::cast<py::tuple>(trajectory[j]);
            const py::tuple key = py::make_tuple(p0[1], p1[1]);
            PyObject *existing = PyDict_GetItemWithError(registry.ptr(), key.ptr());
            if (existing == nullptr && PyErr_Occurred()) {
                throw py::error_already_set();
            }
            size_t index;
            if (existing != nullptr) {
                index = py::cast<size_t>(py::handle(existing));
            } else {
                index = buckets.size();
                registry[key] = index;
                buckets.emplace_back();
                keys.emplace_back(key);
            }
            Bucket &bucket = buckets[index];
            bucket.idxs.push_back(static_cast<int64_t>(i));
            bucket.t0s.push_back(p0[0].cast<double>());
            bucket.t1s.push_back(p1[0].cast<double>());
            p0 = p1;
        }
    }

    py::dict out;
    for (size_t k = 0; k < buckets.size(); ++k) {
        const Bucket &bucket = buckets[k];
        out[keys[k]] = py::make_tuple(to_array<int64_t>(bucket.idxs),
                                      to_time_array(bucket.t0s, float64),
                                      to_time_array(bucket.t1s, float64));
    }
    return out;
}

class LinkBuckets {
  public:
    explicit LinkBuckets(const py::list &link_keys)
        : link_keys_(link_keys), buckets_(py::len(link_keys)),
          used_(py::len(link_keys), 0) {
        for (const py::handle key_h : link_keys) {
            const py::tuple key = py::cast<py::tuple>(key_h);
            dest_states_.emplace_back(py::reinterpret_borrow<py::object>(key[1]));
            alt_map_[py::reinterpret_borrow<py::object>(key[0])].push_back(
                dest_states_.size() - 1);
        }
    }

    void add(py::ssize_t i, const py::handle &s0, double t0, double t1,
             const py::object *s1) {
        const auto it = alt_map_.find(py::reinterpret_borrow<py::object>(s0));
        if (it == alt_map_.end()) {
            return;
        }
        for (const size_t k : it->second) {
            if (!used_[k]) {
                used_[k] = 1;
                order_.push_back(k);
            }
            Bucket &bucket = buckets_[k];
            bucket.idxs.push_back(static_cast<int64_t>(i));
            bucket.t0s.push_back(t0);
            bucket.t1s.push_back(t1);
            bucket.obs.push_back(s1 != nullptr && dest_states_[k].equal(*s1));
        }
    }

    void add_tail(py::ssize_t i, const py::tuple &last, double c_i) {
        const double last_t = last[0].cast<double>();
        if (last_t < c_i) {
            add(i, last[1], last_t, c_i, nullptr);
        }
    }

    template <typename F> py::dict collect(F &&make_value) const {
        py::dict out;
        for (const size_t k : order_) {
            out[link_keys_[static_cast<py::ssize_t>(k)]] = make_value(buckets_[k]);
        }
        return out;
    }

  private:
    const py::list &link_keys_;
    std::unordered_map<py::object, std::vector<size_t>, PyObjHash, PyObjEq>
        alt_map_;
    std::vector<py::object> dest_states_;
    std::vector<Bucket> buckets_;
    std::vector<uint8_t> used_;
    std::vector<size_t> order_;
};

py::dict _build_quad_buckets(const py::list &trajectories, const py::list &link_keys,
                             const py::list &censoring, bool float64) {
    LinkBuckets buckets(link_keys);
    const py::ssize_t n = py::len(trajectories);
    for (py::ssize_t i = 0; i < n; ++i) {
        const py::list trajectory = checked_trajectory(trajectories, i);
        const py::ssize_t m = py::len(trajectory);
        py::tuple p0 = py::cast<py::tuple>(trajectory[0]);
        for (py::ssize_t j = 1; j < m; ++j) {
            const py::tuple p1 = py::cast<py::tuple>(trajectory[j]);
            const py::object s1 = p1[1];
            buckets.add(i, p0[1], p0[0].cast<double>(), p1[0].cast<double>(), &s1);
            p0 = p1;
        }
        buckets.add_tail(i, p0, censoring[i].cast<double>());
    }

    return buckets.collect([float64](const Bucket &bucket) {
        return py::make_tuple(to_array<int64_t>(bucket.idxs),
                              to_time_array(bucket.t0s, float64),
                              to_time_array(bucket.t1s, float64),
                              to_array<bool>(bucket.obs));
    });
}

py::dict _build_remaining_buckets(const py::list &trajectories,
                                  const py::list &link_keys,
                                  const py::list &censoring, bool float64) {
    LinkBuckets buckets(link_keys);
    const py::ssize_t n = py::len(trajectories);
    for (py::ssize_t i = 0; i < n; ++i) {
        const py::list trajectory = checked_trajectory(trajectories, i);
        buckets.add_tail(i, py::cast<py::tuple>(trajectory[py::len(trajectory) - 1]),
                         censoring[i].cast<double>());
    }

    return buckets.collect([float64](const Bucket &bucket) {
        return py::make_tuple(to_array<int64_t>(bucket.idxs),
                              to_time_array(bucket.t0s, float64));
    });
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
