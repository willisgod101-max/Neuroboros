"""
Utility functions for core functionality.
"""
import os
import sys

def enable_utf8_console():
    """
    Enable UTF-8 output in Windows console for proper character display.
    """
    if os.name == 'nt':
        os.system('chcp 65001 > nul')
        # Also reconfigure stdout for UTF-8
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')

