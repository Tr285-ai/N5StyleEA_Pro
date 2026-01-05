import numpy as np
from numba import njit, prange
import socket
import struct
import threading
from queue import Queue, Empty
import time
from collections import defaultdict
import pandas as pd
import psutil
import os

class MarketDataFeed:
    def __init__(self, config: dict):
        self.config = config
        self.callbacks = []
        self.running = False
        self.thread = None
        self.socket = None
        self.buffer = bytearray(65536)  # 64KB buffer
        self.buffer_view = memoryview(self.buffer)
        self.buffer_pos = 0
        self.sequence_number = 0
        self.last_seq_num = {}
        self.dropped_packets = 0
        self.total_packets = 0
        self.latency_stats = defaultdict(list)
        
        # Performance tuning
        self.enable_jit = config.get('enable_jit', True)
        self.batch_size = config.get('batch_size', 100)
        
        # Initialize network
        self.setup_network()
    
    def setup_network(self):
        """Initialize low-latency network connection."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Increase receive buffer size
            rcvbuf = 1024 * 1024 * 100  # 100MB
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
            
            # Bind to multicast group if specified
            if 'multicast_group' in self.config:
                self.socket.bind(('', self.config['port']))
                
                # Add multicast group membership
                group = socket.inet_aton(self.config['multicast_group'])
                mreq = struct.pack('4sL', group, socket.INADDR_ANY)
                self.socket.setsockopt(
                    socket.IPPROTO_IP, 
                    socket.IP_ADD_MEMBERSHIP, 
                    mreq
                )
            else:
                self.socket.bind(('0.0.0.0', self.config['port']))
                
            # Set non-blocking
            self.socket.setblocking(False)
            
        except Exception as e:
            print(f"Network setup error: {e}")
    
    def start(self):
        """Start the market data feed in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the market data feed."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _run(self):
        """Main processing loop."""
        batch = []
        last_stats = time.time()
        
        while self.running:
            try:
                # Receive data
                try:
                    nbytes, addr = self.socket.recvfrom_into(self.buffer_view[self.buffer_pos:])
                    if nbytes > 0:
                        self.buffer_pos += nbytes
                        self.total_packets += 1
                except BlockingIOError:
                    pass
                
                # Process complete messages
                processed = self._process_buffer()
                
                # Update batch
                batch.extend(processed)
                
                # Process batch if full or timeout
                now = time.time()
                if len(batch) >= self.batch_size or (now - last_stats) > 0.001:  # 1ms
                    if batch:
                        self._process_batch(batch)
                        batch = []
                    last_stats = now
                
                # Print stats every second
                if (now - last_stats) >= 1.0:
                    self._print_stats()
                    last_stats = now
                
            except Exception as e:
                print(f"Error in market data feed: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
    
    def _process_buffer(self) -> list:
        """Process data in receive buffer."""
        messages = []
        pos = 0
        
        while pos + 8 <= self.buffer_pos:  # Minimum message size is 8 bytes
            # Parse message header (example format)
            msg_type = int.from_bytes(self.buffer[pos:pos+2], 'big')
            msg_len = int.from_bytes(self.buffer[pos+2:pos+4], 'big')
            
            # Check if we have a complete message
            if pos + msg_len > self.buffer_pos:
                break
                
            # Process complete message
            msg_data = self.buffer[pos:pos+msg_len]
            messages.append((msg_type, msg_data, time.time_ns()))
            
            # Move to next message
            pos += msg_len
        
        # Remove processed data from buffer
        if pos > 0:
            self.buffer = self.buffer[pos:]
            self.buffer_pos -= pos
            
        return messages
    
    def _process_batch(self, messages: list):
        """Process a batch of market data messages."""
        if self.enable_jit:
            self._process_batch_jit(messages)
        else:
            for msg_type, msg_data, timestamp in messages:
                self._process_single_message(msg_type, msg_data, timestamp)
    
    @staticmethod
    @njit(parallel=True)
    def _process_batch_jit(messages):
        """JIT-accelerated batch processing."""
        # This is a placeholder - in practice, you'd implement
        # the message processing logic here using numba-compatible code
        for i in prange(len(messages)):
            pass
    
    def _process_single_message(self, msg_type: int, msg_data: bytes, timestamp: int):
        """Process a single market data message."""
        try:
            # Process based on message type
            if msg_type == 1:  # Price update
                symbol = msg_data[4:12].decode('ascii').strip()
                price = struct.unpack('!d', msg_data[12:20])[0]
                size = struct.unpack('!d', msg_data[20:28])[0]
                
                # Update order book
                self._update_order_book(symbol, price, size, msg_data[28] == 1)
                
                # Record latency
                latency_ns = time.time_ns() - timestamp
                self.latency_stats['price_update'].append(latency_ns)
                
            elif msg_type == 2:  # Trade
                # Process trade message
                pass
                
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def _update_order_book(self, symbol: str, price: float, size: float, is_bid: bool):
        """Update order book with new price level."""
        # This would update an order book data structure
        # and trigger callbacks
        for callback in self.callbacks:
            try:
                callback(symbol, price, size, is_bid)
            except Exception as e:
                print(f"Error in callback: {e}")
    
    def register_callback(self, callback):
        """Register a callback for market data updates."""
        self.callbacks.append(callback)
    
    def _print_stats(self):
        """Print performance statistics."""
        if self.total_packets > 0:
            drop_rate = (self.dropped_packets / self.total_packets) * 100
        else:
            drop_rate = 0.0
            
        print(f"Packets: {self.total_packets}/s, "
              f"Dropped: {self.dropped_packets} ({drop_rate:.2f}%)")
        
        # Print latency stats
        for metric, values in self.latency_stats.items():
            if values:
                arr = np.array(values)
                print(f"{metric} latency (ns): "
                      f"p50={np.percentile(arr, 50):.0f} "
                      f"p95={np.percentile(arr, 95):.0f} "
                      f"p99={np.percentile(arr, 99):.0f}")
                
        # Reset counters
        self.dropped_packets = 0
        self.total_packets = 0
        self.latency_stats.clear()

# Optimized data structures
class CircularBuffer:
    """Lock-free circular buffer for high-performance data storage."""
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float64)
        self.head = 0
        self.tail = 0
        self.count = 0
    
    def push(self, value: float) -> bool:
        """Add a value to the buffer."""
        if self.count >= self.size:
            return False  # Buffer full
            
        self.buffer[self.head] = value
        self.head = (self.head + 1) % self.size
        self.count += 1
        return True
    
    def pop(self) -> float:
        """Remove and return the oldest value."""
        if self.count <= 0:
            raise IndexError("Buffer is empty")
            
        value = self.buffer[self.tail]
        self.tail = (self.tail + 1) % self.size
        self.count -= 1
        return value
    
    def clear(self):
        """Clear the buffer."""
        self.head = 0
        self.tail = 0
        self.count = 0

# Example usage
if __name__ == "__main__":
    # System optimization
    import os
    import psutil
    
    p = psutil.Process()
    p.cpu_affinity([0])  # Pin to first CPU core
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    
    # Initialize market data feed
    config = {
        'host': '0.0.0.0',
        'port': 5000,
        'multicast_group': '239.255.0.1',
        'enable_jit': True,
        'batch_size': 100
    }
    
    feed = MarketDataFeed(config)
    
    # Example callback
    def on_market_data(symbol, price, size, is_bid):
        print(f"{symbol} {'BID' if is_bid else 'ASK'}: {price} @ {size}")
    
    feed.register_callback(on_market_data)
    
    try:
        print("Starting market data feed...")
        feed.start()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        feed.stop()
