"""
Process forking code copied from https://github.com/psathyrella/partis/blob/master/python/utils.py
"""

import time
import sys
import os
from subprocess import check_output, Popen

class CustomCommand:
    def __init__(self, cmd_str, outfname, logdir, env, threads=None):
        self.cmd_str = cmd_str
        self.outfname = outfname
        self.logdir = logdir
        self.env = env
        self.threads = threads

# bash color codes
Colors = {}
Colors['head'] = '\033[95m'
Colors['bold'] = '\033[1m'
Colors['purple'] = '\033[95m'
Colors['blue'] = '\033[94m'
Colors['light_blue'] = '\033[1;34m'
Colors['green'] = '\033[92m'
Colors['yellow'] = '\033[93m'
Colors['red'] = '\033[91m'
Colors['reverse_video'] = '\033[7m'
Colors['end'] = '\033[0m'

def color(col, seq, width=None, padside='left'):
    return_str = [Colors[col], seq, Colors['end']]
    if width is not None:  # make sure final string prints to correct width
        n_spaces = max(0, width - len(seq))  # if specified <width> is greater than uncolored length of <seq>, pad with spaces so that when the colors show up properly the colored sequences prints with width <width>
        if padside == 'left':
            return_str.insert(0, n_spaces * ' ')
        elif padside == 'right':
            return_str.insert(len(return_str), n_spaces * ' ')
        else:
            assert False
    return ''.join(return_str)

def run_cmd(cmdfo, batch_system=None, batch_options=None):
    """
    Creates a process command

    @param cmdfo: class CustomCommand
    @param batch_system: "slurm" or "sge" -- right now we only accept slurm
    @param batch_options: other options to pass to the batch system manager
    """
    cmd_str = cmdfo.cmd_str  # don't want to modify the str in <cmdfo>

    fout = cmdfo.logdir + '/out.txt'
    ferr = cmdfo.logdir + '/err.txt'
    prefix = None
    if batch_system is not None:
        if batch_system == 'slurm':
            prefix = 'srun -p restart,matsen_e,campus --exclude=data/gizmod.txt'
            if cmdfo.threads is not None:
                prefix += ' --cpus-per-task %d' % cmdfo.threads
        elif batch_system == 'sge':
            prefix = 'qsub -sync y -b y -V -o ' + fout + ' -e ' + ferr
            fout = None
            ferr = None
        else:
            assert False
        if batch_options is not None:
            prefix += ' ' + batch_options
    if prefix is not None:
        cmd_str = prefix + ' ' + cmd_str

    if not os.path.exists(cmdfo.logdir):
        os.makedirs(cmdfo.logdir)

    if fout is not None and ferr is not None:
        fout_f = open(fout, 'w')
        ferr_f = open(ferr, 'w')
        proc = Popen(cmd_str.split(),
                     stdout=fout_f,
                     stderr=ferr_f,
                     env=cmdfo.env)
        fout_f.close()
        ferr_f.close()
    else:
        proc = Popen(cmd_str.split(),
                     stdout=None,
                     stderr=None,
                     env=cmdfo.env)

    #print "Running process:", cmd_str
    return proc

def run_cmds(cmdfos, sleep=True, batch_system="slurm", batch_options=None, debug=None):
    """
    Kick off processes to the batch system
    NOTE: This will try a number of times, and if that fails, this function will NOT throw an error. Instead, if the output file is not there,
    it is up to you to figure out what to do!

    @param cmdfos: list of CustomCommands
    @param sleep: Whether to sleep between adding processes. Set sleep to False if you're commands are going to run really really really quickly
    @param batch_system: "slurm" or "sge" -- right now it should only be "slurm"
    @param batch_options: other options to pass to the batch system manager
    @param debug: None - don't print things unless error, "print" - print things, "write" - write logs to file
    """
    procs, n_tries = [], []
    for iproc in range(len(cmdfos)):
        procs.append(run_cmd(cmdfos[iproc], batch_system=batch_system, batch_options=batch_options))
        n_tries.append(1)
        if sleep:
            time.sleep(0.01)
    while procs.count(None) != len(procs):  # we set each proc to None when it finishes
        for iproc in range(len(cmdfos)):
            if procs[iproc] is None:  # already finished
                continue
            if procs[iproc].poll() is not None:  # it just finished
                finish_process(iproc, procs, n_tries, cmdfos[iproc], batch_system=batch_system, batch_options=batch_options, debug=debug)
        sys.stdout.flush()
        if sleep:
            time.sleep(0.01)

def finish_process(iproc, procs, n_tries, cmdfo, batch_system=None, batch_options=None, debug=None, max_num_tries=2):
    """
    Deal with a process once it's finished (i.e. check if it failed, and restart if so)
    """
    procs[iproc].communicate() # send data to stdin, read data from stdout and stderr, wait for process to terminate
    if procs[iproc].returncode == 0:
        if os.path.exists(cmdfo.outfname):
            process_out_err('', '', extra_str='' if len(procs) == 1 else str(iproc), logdir=cmdfo.logdir, debug=debug)
            procs[iproc] = None  # job succeeded
            return
        else:
            print('      proc %d succeded but its output isn\'t there, so sleeping for a bit...' % iproc)
            for i in range(30):
                if os.path.exists(cmdfo.outfname):
                    process_out_err('', '', extra_str='' if len(procs) == 1 else str(iproc), logdir=cmdfo.logdir, debug=debug)
                    procs[iproc] = None  # job succeeded
                    return
                time.sleep(1)
    # handle failure
    if n_tries[iproc] > max_num_tries:
        # Time to give up!
        print('exceeded max number of tries for cmd\n    %s\nlook for output in %s' % (cmdfo.cmd_str, cmdfo.logdir))
        procs[iproc] = None  # job failed but we gave up
        return
    else:
        print('    proc %d try %d' % (iproc, n_tries[iproc]))
        if procs[iproc].returncode == 0 and not os.path.exists(cmdfo.outfname):  # don't really need both the clauses
            print('succeded but output is missing')
        else:
            print('failed with %d (output %s)' % (procs[iproc].returncode, 'exists' if os.path.exists(cmdfo.outfname) else 'is missing'))
        for strtype in ['out.txt', 'err.txt']:
            if os.path.exists(cmdfo.logdir + '/' + strtype) and os.stat(cmdfo.logdir + '/' + strtype).st_size > 0:
                print('        %s tail:' % strtype)
                logstr = check_output(['tail', cmdfo.logdir + '/' + strtype])
                print('\n'.join(['            ' + l for l in logstr.split('\n')]))
        if batch_system is not None and os.path.exists(cmdfo.logdir + '/err.txt'):  # cmdfo.cmd_str.split()[0] == 'srun' and
            jobid = ''
            try:
                jobid = check_output(['head', '-n1', cmdfo.logdir + '/err.txt']).split()[2]
                nodelist = check_output(['squeue', '--job', jobid, '--states=all', '--format', '%N']).split()[1]
            except:
                print('      couldn\'t get node list for jobid \'%s\'' % jobid)
            # try:
            #     print '        sshing to %s' % nodelist
            #     outstr = check_output('ssh -o StrictHostKeyChecking=no ' + nodelist + ' ps -eo pcpu,pmem,rss,cputime:12,stime:7,user,args:100 --sort pmem | tail', shell=True)
            #     print pad_lines(outstr, padwidth=12)
            # except:
            #     print '        failed'
        print('    restarting proc %d' % iproc)
        procs[iproc] = run_cmd(cmdfo, batch_system=batch_system, batch_options=batch_options)
        n_tries[iproc] += 1

def process_out_err(out, err, extra_str='', logdir=None, debug=None):
    """
    Process the out and err files. Any lines with the string "force" will be printed out.
    This will also delete all the out and err files.
    """
    if logdir is not None:
        def readfile(fname):
            try:
                ftmp = open(fname)
                fstr = ''.join(ftmp.readlines())
                ftmp.close()
                os.remove(fname)
            except Exception as e:
                print("Warning process_out_err %s" % e)
                fstr = ''
            return fstr
        out = readfile(logdir + '/out.txt')
        err = readfile(logdir + '/err.txt')

    for line in out.split('\n'):  # temporarily (maybe) print debug info realted to --n-final-clusters/force merging
        if 'force' in line:
            print('    %s %s' % (color('yellow', 'force info:'), line))

    err_str = ''
    for line in err.split('\n'):
        if 'stty: standard input: Inappropriate ioctl for device' in line:
            continue
        if 'srun: job' in line and 'queued and waiting for resources' in line:
            continue
        if 'srun: job' in line and 'has been allocated resources' in line:
            continue
        if 'GSL_RNG_TYPE=' in line or 'GSL_RNG_SEED=' in line:
            continue
        if '[ig_align] Read' in line or '[ig_align] Aligned' in line:
            continue
        if len(line.strip()) > 0:
            err_str += line + '\n'

    if debug is None:
        if err_str != '':
            print(err_str)
    elif err_str + out != '':
        if debug == 'print':
            if extra_str != '':
                print('      --> proc %s' % extra_str)
            print(err_str + out)
        elif debug == 'write':
            logfile_name = logdir + '/log.txt'
            print('writing log to %s' % logfile_name)
            with open(logfile_name, 'w') as logfile:
                logfile.write(err_str + out)
        else:
            assert False
