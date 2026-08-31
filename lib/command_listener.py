import socket
import threading
import queue

from logging_utils import *

class CommandListener:
    def __init__(self, buffer_size=16):
        self.buffer_size = buffer_size
        self.queue = queue.Queue()
        self.server_socket = None
        self.is_running = False
        self.thread = None

    def start(self, host='0.0.0.0', port=5555):
        if self.is_running:
            Logging.get().warn('CommandListener is already running')
            return

        self.is_running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)

        # Run as a daemon thread so it dies automatically if the main script crashes
        self.thread = threading.Thread(target=self.listen_loop, daemon=True)
        self.thread.start()

    def listen_loop(self):
        while self.is_running:
            try:
                conn, addr = self.server_socket.accept()
            except Exception:
                # Triggers when socket is closed via stop()
                assert not self.is_running
                Logging.get().debug('CommandListener goes down')
                break

            try:
                self.handle_connection(conn, addr)
            except Exception as e:
                Logging.get().error(f'Failed to handle connection from {addr}: {e}')
            finally:
                conn.close()

    def handle_connection(self, conn, addr):
        buffer = ''
        
        while self.is_running:
            chunk = conn.recv(self.buffer_size).decode('utf-8', errors='ignore')
            
            if not chunk:
                # Client disconnected before sending a full line
                break
            
            buffer += chunk
            
            if '\n' in buffer:
                # Process the message up to the first newline
                message, _, remainder = buffer.partition('\n')
                command = message.strip().lower()
                
                if command:
                    Logging.get().debug(f'Received command from {addr}: {command}')
                    self.queue.put(command)
                    conn.sendall(b"OK\n")
                
                # Keep remainder in case multiple commands were packed together
                buffer = remainder
                break

    def get_command(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        if not self.is_running:
            return
        
        Logging.get().debug('Stopping CommandListener')
        self.is_running = False
        
        if self.server_socket:
            try:
                # Forces accept() to unblock and throw an exception, exiting the thread safely
                self.server_socket.close()
            except Exception:
                pass
        
        if self.thread:
            self.thread.join(timeout=2.0)
