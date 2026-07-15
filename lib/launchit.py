import os
import sys
import json
import re
import string
from collections import namedtuple, defaultdict
from dataclasses import dataclass
from enum import IntEnum, auto 

import lang_utils as lu
from logging_utils import *

class Command(IntEnum):
    COLLECT = auto()
    COLLECTED = auto()
    DISABLE = auto()
    STOP = auto()

@dataclass(order=True, slots=True)
class ExecGraphEntry:
    command: object = None
    cell_ind: int = None
    source_line_ind: int = None
    is_oneliner: bool = None
    stop_source_line_ind: int = None
    index: str = None

class NotebookProcessor:
    def __init__(self):
        self.nb = None
        self.exec_graph = None
        self.collected_source_lines = None

    def __call__(self, f, new_fname, expandvars, collect_inds, disable_inds):
        self.nb = json.load(f)
        self.exec_graph = []
        self.collected_source_lines = defaultdict(list) # collect index -> source lines
        
        for cell_ind, cell in enumerate(self.nb['cells']):
            for source_line_ind, source_line in enumerate(cell['source']):
                source_line = source_line.strip()
                m = re.match(r'^(.*)#\s*@launchit\.(\w+)\s*$', source_line)
                
                if m:
                    Logging.trace(f'Cell {cell_ind}, launchit stanza: "{source_line}"')
                    before = m.group(1)
                    command_with_index = m.group(2)
                    m2 = re.match(r'^(collected|collect|disable|stop)(_(\w+))?$', command_with_index)

                    if not m2:
                        Logging.warn(f'WARNING! Cell {cell_ind} contains unrecognized launchit command: "{command_with_index}"')
                    else:
                        command = m2.group(1)
                        index = m2.group(3)
                        ege = ExecGraphEntry(
                            command=Command[command.upper()], 
                            cell_ind=cell_ind, 
                            source_line_ind=source_line_ind, 
                            is_oneliner=re.match(r'[^\s]+', before), 
                            stop_source_line_ind=-1,
                            index=index,
                        )

                        if ege.command == Command.COLLECTED:
                            assert not ege.is_oneliner, '@launchit.collected cannot be oneliner'
                        elif ege.command == Command.STOP:
                            assert not ege.is_oneliner, '@launchit.stop cannot be oneliner'
                            assert index is None, f'@launchit.stop does not support indexing, {command_with_index=}'
                            # look behind and patch stop_source_line_ind
                            for lb_ege_ind in range(len(self.exec_graph) - 1, -1, -1):
                                lb_ege = self.exec_graph[lb_ege_ind]
                                
                                if lb_ege.cell_ind != cell_ind:
                                    raise Exception(f'Cell {cell_ind}, line {source_line_ind}, @launchit.stop has no preceeding command')
                                elif lb_ege.stop_source_line_ind == -1:
                                    self.exec_graph[lb_ege_ind].stop_source_line_ind = source_line_ind
                                    Logging.trace(f'Cell {cell_ind}, command {lb_ege.command.name} at line {lb_ege.source_line_ind} will stop at line {source_line_ind}')
                                    break

                        if ege.command != Command.STOP:
                            self.exec_graph.append(ege)

        for ege in sorted(self.exec_graph):
            cell = self.nb['cells'][ege.cell_ind]
            stop_source_line_ind = len(cell['source']) if ege.stop_source_line_ind == -1 else ege.stop_source_line_ind
            
            match ege.command:
                case Command.COLLECT:
                    do_collect = collect_inds is None # we are asked to grab everything

                    if not do_collect:
                        do_collect = ege.index is None # wildcard collect instruction

                    if not do_collect:
                        assert ege.index is not None
                        do_collect = ege.index in collect_inds or lu.from_str(int, ege.index, ege.index) in collect_inds # old clients expect just integer indices

                    if not do_collect:
                        if ege.is_oneliner:
                            Logging.trace(f'Cell {ege.cell_ind}, skip collecting source line {ege.source_line_ind}, index={ege.index}')
                        else:
                            assert ege.source_line_ind + 1 < stop_source_line_ind
                            Logging.trace(f'Cell {ege.cell_ind}, skip collecting source lines from {ege.source_line_ind + 1} to {stop_source_line_ind}, index={ege.index}')
                    else:
                        if ege.is_oneliner:
                            Logging.trace(f'Cell {ege.cell_ind}, collecting source line {ege.source_line_ind}, index={ege.index}')
                            source_line = cell['source'][ege.source_line_ind]
                            self.collected_source_lines[ege.index].append(source_line)
                        else:
                            assert ege.source_line_ind + 1 < stop_source_line_ind
                            Logging.trace(f'Cell {ege.cell_ind}, collecting source lines from {ege.source_line_ind + 1} to {stop_source_line_ind}, index={ege.index}')
        
                            if self.collected_source_lines:
                                self.collected_source_lines[ege.index].append('\n')
            
                            for source_line_ind in range(ege.source_line_ind + 1, stop_source_line_ind):
                                source_line = cell['source'][source_line_ind]
                                self.collected_source_lines[ege.index].append(source_line)
                
                case Command.COLLECTED:
                    my_collected_source_lines = self.collected_source_lines.get(ege.index, [])
                    my_collected_source_lines.append('')
                    
                    Logging.trace(f'Cell {ege.cell_ind} (len={len(cell['source'])}), ' + 
                                  f'putting {len(my_collected_source_lines)} collected source lines to {ege.source_line_ind}, index={ege.index}')
    
                    for ind, source_line in enumerate(my_collected_source_lines):
                        if ind > 0:
                            cell['source'].insert(ege.source_line_ind + ind, source_line)
                        else:
                            cell['source'][ege.source_line_ind + ind] = source_line

                    successors = filter(
                        lambda x: x.command == Command.COLLECTED and x.cell_ind == ege.cell_ind and x.source_line_ind > ege.source_line_ind,
                        self.exec_graph,
                    )
                    
                    for successor in successors:
                        successor.source_line_ind += max(len(my_collected_source_lines) - 1, 0) # -1 since collected instruction is replaced with first line
            
                case Command.DISABLE:
                    def disable_source_line(s):
                        do_disable = disable_inds is None # we asked to disable everything

                        if not do_disable:
                            do_disable = ege.index is None # wildcard disable instruction

                        if not do_disable:
                            assert ege.index is not None
                            do_disable = ege.index in disable_inds or lu.from_str(int, ege.index, ege.index) in disable_inds # old clients expect just integer indices

                        if not do_disable:
                            return s
                            
                        if re.match(r'^\s*#', s):
                            return s # already disabled
                        else:
                            return '# ' + s
                    
                    if ege.is_oneliner:
                        Logging.trace(f'Cell {ege.cell_ind}, disabling source line {ege.source_line_ind}, index={ege.index}')
                        cell['source'][ege.source_line_ind] = disable_source_line(cell['source'][ege.source_line_ind])
                    else:
                        assert ege.source_line_ind + 1 < stop_source_line_ind
                        Logging.trace(f'Cell {ege.cell_ind}, disabling source lines from {ege.source_line_ind + 1} to {stop_source_line_ind}, index={ege.index}')
    
                        for source_line_ind in range(ege.source_line_ind + 1, stop_source_line_ind):
                            cell['source'][source_line_ind] = disable_source_line(cell['source'][source_line_ind])
                case _:
                    assert False, f'Failed to understand exec_graph_entry={ege}'

        expandvars['LAUNCHIT_FNAME'] = new_fname
        Logging.trace(f'{expandvars=}')
        
        for cell in self.nb['cells']:
            for source_line_ind, source_line in enumerate(cell['source']):
                t = string.Template(source_line)
                cell['source'][source_line_ind] = t.safe_substitute(expandvars)

    
def launchit(fname, launch_serial=0, expandvars={}, make_py_file=False, dir_name='', max_serials_count=1_000, collect_inds=None, disable_inds=None):
    fname_dir = os.path.dirname(fname) if not dir_name else dir_name
    fname_name = os.path.splitext(os.path.basename(fname))[0]
    fname_ext = os.path.splitext(fname)[1] if not make_py_file else '.py'
    
    new_fname = ''
    serials_range = range(1, max_serials_count + 1) if launch_serial == 0 else [launch_serial]

    for i in serials_range:
        new_fname = os.path.join(fname_dir, f'{fname_name}-launch{i}{fname_ext}')
    
        if not os.path.exists(new_fname):
            break
    else:
        raise Exception(f'Failed to generate new launch file name: all variants are taken')

    Logging.debug(f'Creating {new_fname}')
    
    processor = NotebookProcessor()
    
    with open(fname, 'r') as f:
        processor(f, new_fname=new_fname, expandvars=expandvars, collect_inds=collect_inds, disable_inds=disable_inds)
    
    with open(new_fname, 'w') as f:
        if make_py_file:
            for cell in processor.nb['cells']:
                if cell['cell_type'] == 'code':
                    f.writelines(cell['source'])
                    f.write('\n\n')
        else:
            json.dump(processor.nb, f, indent=2) # .ipynb
    
    return new_fname

def extract_source_code(fname, collect_inds=[]):
    processor = NotebookProcessor()

    with open(fname, 'rt') as f:
        processor(f, fname, expandvars={}, collect_inds=collect_inds, disable_inds=[])

    if collect_inds is None or not collect_inds:
        return '\n'.join(processor.collected_source_lines[None])

    result = ''
        
    for i in collect_inds:
        result += '\n'.join(processor.collected_source_lines[i])

    return result
