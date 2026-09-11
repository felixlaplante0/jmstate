// Survival bucket construction for jmstate.
//
// Single-pass grouping of multistate trajectories. States are arbitrary
// Python hashables, so grouping keys stay Python objects and only the tight
// per-segment loops move to C++. Numeric columns are returned as plain nested
// vectors; the Python wrapper materializes them as torch tensors with the
// requested dtype and device in one call per column.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Accumulators for one transition key.
struct Bucket {
    std::vector<int64_t> idxs;
    std::vector<double> t0s;
    std::vector<double> t1s;
    std::vector<uint8_t> obs;
};

// Finds or creates the bucket index for a key in a dict-backed registry.
py::ssize_t find_or_create(py::dict &registry, std::vector<Bucket> &buckets,
                           std::vector<py::object> &keys, const py::tuple &key) {
    if (registry.contains(key)) {
        return py::cast<py::ssize_t>(registry[key]);
    }
    const py::ssize_t index = static_cast<py::ssize_t>(buckets.size());
    registry[key] = index;
    buckets.emplace_back();
    keys.emplace_back(py::reinterpret_borrow<py::object>(key));
    return index;
}

// Groups observed segments by (from_state, to_state).
//
// Returns a dict mapping key tuples to (idxs, t0s, t1s) Python lists.
py::dict build_buckets_raw(const py::list &trajectories) {
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
        for (py::ssize_t j = 0; j + 1 < m; ++j) {
            const py::tuple p0 = trajectory[j];
            const py::tuple p1 = trajectory[j + 1];
            const double t0 = py::float_(p0[0]);
            const double t1 = py::float_(p1[0]);
            const py::tuple key = py::make_tuple(p0[1], p1[1]);
            const py::ssize_t index = find_or_create(registry, buckets, keys, key);
            Bucket &bucket = buckets[static_cast<size_t>(index)];
            bucket.idxs.push_back(static_cast<int64_t>(i));
            bucket.t0s.push_back(t0);
            bucket.t1s.push_back(t1);
        }
    }

    py::dict out;
    for (size_t k = 0; k < buckets.size(); ++k) {
        const Bucket &bucket = buckets[k];
        out[keys[k]] = py::make_tuple(bucket.idxs, bucket.t0s, bucket.t1s);
    }
    return out;
}

// Builds the state -> link-key-indices lookup shared by quad/remaining buckets.
void build_alt_map(const py::list &link_keys, py::dict &alt_map,
                   std::vector<py::object> &dest_states) {
    dest_states.reserve(static_cast<size_t>(py::len(link_keys)));
    for (const auto &key_h : link_keys) {
        const py::tuple key = py::cast<py::tuple>(key_h);
        const py::ssize_t k = static_cast<py::ssize_t>(dest_states.size());
        dest_states.emplace_back(py::reinterpret_borrow<py::object>(key[1]));
        const py::object src = py::reinterpret_borrow<py::object>(key[0]);
        if (!alt_map.contains(src)) {
            alt_map[src] = py::list();
        }
        alt_map[src].attr("append")(k);
    }
}

// Vectorizable buckets: observed segments expanded over competing transitions
// plus censored tails. Returns key tuples to (idxs, t0s, t1s, obs) lists.
py::dict build_quad_buckets_raw(const py::list &trajectories, const py::list &link_keys,
                                const py::list &censoring) {
    py::dict alt_map;
    std::vector<py::object> dest_states;
    build_alt_map(link_keys, alt_map, dest_states);

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
        for (py::ssize_t j = 0; j + 1 < m; ++j) {
            const py::tuple p0 = trajectory[j];
            const py::tuple p1 = trajectory[j + 1];
            const double t0 = py::float_(p0[0]);
            const double t1 = py::float_(p1[0]);
            const py::object s0 = py::reinterpret_borrow<py::object>(p0[1]);
            const py::object s1 = py::reinterpret_borrow<py::object>(p1[1]);
            if (!alt_map.contains(s0)) {
                continue;
            }
            for (const auto &k_h : alt_map[s0]) {
                const py::ssize_t k = py::cast<py::ssize_t>(k_h);
                const py::tuple key = link_keys[k];
                const py::ssize_t index = find_or_create(registry, buckets, keys, key);
                Bucket &bucket = buckets[static_cast<size_t>(index)];
                bucket.idxs.push_back(static_cast<int64_t>(i));
                bucket.t0s.push_back(t0);
                bucket.t1s.push_back(t1);
                bucket.obs.push_back(dest_states[static_cast<size_t>(k)].equal(s1));
            }
        }
        const py::tuple last = trajectory[m - 1];
        const double last_t = py::float_(last[0]);
        const double c_i = py::float_(censoring[i]);
        if (last_t >= c_i) {
            continue;
        }
        const py::object last_s = py::reinterpret_borrow<py::object>(last[1]);
        if (!alt_map.contains(last_s)) {
            continue;
        }
        for (const auto &k_h : alt_map[last_s]) {
            const py::ssize_t k = py::cast<py::ssize_t>(k_h);
            const py::tuple key = link_keys[k];
            const py::ssize_t index = find_or_create(registry, buckets, keys, key);
            Bucket &bucket = buckets[static_cast<size_t>(index)];
            bucket.idxs.push_back(static_cast<int64_t>(i));
            bucket.t0s.push_back(last_t);
            bucket.t1s.push_back(c_i);
            bucket.obs.push_back(false);
        }
    }

    py::dict out;
    for (size_t k = 0; k < buckets.size(); ++k) {
        const Bucket &bucket = buckets[k];
        out[keys[k]] = py::make_tuple(bucket.idxs, bucket.t0s, bucket.t1s, bucket.obs);
    }
    return out;
}

// Possible buckets: censored tails only. Returns key tuples to (idxs, t0s).
py::dict build_remaining_buckets_raw(const py::list &trajectories,
                                     const py::list &link_keys,
                                     const py::list &censoring) {
    py::dict alt_map;
    std::vector<py::object> dest_states;
    build_alt_map(link_keys, alt_map, dest_states);

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
        const py::tuple last = trajectory[m - 1];
        const double last_t = py::float_(last[0]);
        const double c_i = py::float_(censoring[i]);
        if (last_t >= c_i) {
            continue;
        }
        const py::object last_s = py::reinterpret_borrow<py::object>(last[1]);
        if (!alt_map.contains(last_s)) {
            continue;
        }
        for (const auto &k_h : alt_map[last_s]) {
            const py::tuple key = link_keys[py::cast<py::ssize_t>(k_h)];
            const py::ssize_t index = find_or_create(registry, buckets, keys, key);
            Bucket &bucket = buckets[static_cast<size_t>(index)];
            bucket.idxs.push_back(static_cast<int64_t>(i));
            bucket.t0s.push_back(last_t);
        }
    }

    py::dict out;
    for (size_t k = 0; k < buckets.size(); ++k) {
        const Bucket &bucket = buckets[k];
        out[keys[k]] = py::make_tuple(bucket.idxs, bucket.t0s);
    }
    return out;
}

PYBIND11_MODULE(_surv_ext, m) {
    m.doc() = "C++ survival bucket construction for jmstate.";
    m.def("build_buckets_raw", &build_buckets_raw, py::arg("trajectories"),
          "Group observed segments by (from_state, to_state).");
    m.def("build_quad_buckets_raw", &build_quad_buckets_raw, py::arg("trajectories"),
          py::arg("link_keys"), py::arg("censoring"),
          "Build vectorizable buckets with competing transitions and tails.");
    m.def("build_remaining_buckets_raw", &build_remaining_buckets_raw,
          py::arg("trajectories"), py::arg("link_keys"), py::arg("censoring"),
          "Build censored-tail buckets.");
}
