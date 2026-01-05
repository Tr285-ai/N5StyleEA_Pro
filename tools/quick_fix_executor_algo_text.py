from pathlib import Path

p = Path(__file__).resolve().parent.parent / 'executor.py'
s = p.read_text(encoding='utf-8')
orig = s

# Fix mis-indented log_json under try in _execute_live_algo
s = s.replace(
    "\n                try:\n                log_json('order_filled',",
    "\n                try:\n                    log_json('order_filled',",
)
# Fix mis-indented except line and align pass/self._emit_tca_record
s = s.replace(
    "\n            except Exception:\n                pass\n                self._emit_tca_record({",
    "\n                except Exception:\n                    pass\n                self._emit_tca_record({",
)
# Normalize overly indented session_id key
s = s.replace("\n                                        'session_id': self.session_id,\n",
              "\n                    'session_id': self.session_id,\n")

if s != orig:
    p.write_text(s, encoding='utf-8')
    print('executor.py algo block indentation corrected')
else:
    print('No changes required for executor.py algo block')
