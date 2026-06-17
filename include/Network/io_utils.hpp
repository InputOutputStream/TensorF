#pragma once

#include <unistd.h>
#include <cstddef>
#include <sys/types.h>

/// Read exactly `len` bytes from `fd`, retrying on partial reads.
/// Returns false if the connection drops before all bytes are received.
inline bool read_exact(int fd, void* buf, size_t len) {
    size_t done = 0;
    while (done < len) {
        ssize_t r = ::read(fd, static_cast<char*>(buf) + done, len - done);
        if (r <= 0) return false;
        done += static_cast<size_t>(r);
    }
    return true;
}

/// Write exactly `len` bytes to `fd`, retrying on partial writes.
/// Returns false if the connection drops before all bytes are sent.
inline bool write_exact(int fd, const void* buf, size_t len) {
    size_t done = 0;
    while (done < len) {
        ssize_t w = ::write(fd, static_cast<const char*>(buf) + done, len - done);
        if (w <= 0) return false;
        done += static_cast<size_t>(w);
    }
    return true;
}