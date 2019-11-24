#!/usr/bin/python3

# A Python script to generate training data given ELF or PE files including their
# relevant debug info.
#

import subprocess, os, fnmatch, codecs, random, shutil, multiprocessing, glob
import sys, bisect, numpy, traceback, re
from absl import app
from absl import flags
from subprocess import Popen, PIPE, STDOUT
from operator import itemgetter
from collections import defaultdict

import numpy as np

# Make sure you have a trailing slash.
flags.DEFINE_string('work_directory',
  "/tmp/train_data",
  "The directory into which the training data will be written")

# Generate the fingerprint hashes.
flags.DEFINE_boolean('generate_fingerprints', True, "Decides whether the " +
  "hashes of all features should be extracted and written.")

# Generate full disassemblies in JSON, too. This is not necessary for any of
# the tools in this repository, but may be useful if you wish to experiment
# with disassembly data in other machine-learning contexts.
flags.DEFINE_boolean('generate_json_data', True, "Decides whether JSON output " +
  "of the full disassembly should be extracted and written (not necessary for " +
  "training, but useful for diagnostics and visualization.")

# Disable the use of mnemonic data.
flags.DEFINE_boolean('disable_mnemonic', False, "Disable the extraction of " +
  "mnemonic-based features. Useful to test the power of mnemonics vs. graphs.")

# Clobber existing data directory or not.
flags.DEFINE_boolean('clobber', True, "Clobber output directory or not.")

# Directory for executable files to train on.
flags.DEFINE_string('executable_directory', './',
  "The directory where the ELF and PE executables to train on can be found " +\
  "in their relevant subdirectories ELF/**/* and PE/**/*")

flags.DEFINE_integer('parallelism', 3, "Number of parallel invocations of " +
  "the disassembly tools. Given that one disassembly operation can eat up to " +
  "12GB of RAM for large executables, this is heavily RAM-bound.")
#=============================================================================

FLAGS = flags.FLAGS

def FindELFTrainingFiles():
  """ Returns the list of ELF files that should be used for training. These
  ELF files need to contain objdump-able debug information.

  """
  elf_files = [ filename for filename in glob.iglob(
    FLAGS.executable_directory + 'ELF/**/*', recursive=True)
    if os.path.isfile(filename) ]
  elf_files = elf_files[0:2]
  print("Returning list of files from ELF directory: %s" % elf_files)
  return elf_files

def FindPETrainingFiles():
  """ Returns the list of PE files that should be used for training. These
  PE files need to have associated text files (with suffix .debugdump) that
  contains the output of dia2dump in the same directory. """
  exe_files = [ filename for filename in glob.iglob(
      FLAGS.executable_directory + 'PE/**/*.exe',
    recursive=True) if os.path.isfile(filename) ]
  dll_files = [ filename for filename in glob.iglob(
    FLAGS.executable_directory + 'PE/**/*.dll',
    recursive=True) if os.path.isfile(filename) ]
  print(FLAGS.executable_directory + 'PE/**/*.exe')
  result = exe_files + dll_files
  result = result[0:2]
  print("Returning list of files from PE directory: %s" % result)
  return result

def ObtainFunctionSymbols(training_file, file_format):
  if file_format == "ELF":
    return ObtainELFFunctionSymbols(training_file)
  elif file_format == "PE":
    return ObtainPEFunctionSymbols(training_file)

def SaneBase64(input_string):
  """ Because Python3 attempts to win 'most idiotic language ever', encoding a
  simple string as base64 without risking to have strange newlines added is
  difficult. This functions is an insane solution: Call command line
  base64encode instead of dealing with Python. """
  encoded_string = subprocess.run(["base64", "-w0"], stdout=PIPE,
    input=bytes(input_string, encoding="utf-8")).stdout.decode("utf-8")
  return encoded_string

def ObtainELFFunctionSymbols(training_file):
  """ Runs objdump to obtain the symbols in an ELF file and then returns a
  dictionary for this file. """
  result = {}
  symbols = [ line for line in subprocess.check_output(
    [ "objdump", "-t", training_file ] ).decode("utf-8").split("\n")
      if line.find(" F .text") != -1 ]
  syms_and_address = []
  for sym in symbols:
    tokens = sym.split()
    if tokens[2] == 'F':
      address = int(tokens[0], 16)
      # Run the string through c++filt
      sym = subprocess.check_output([ "c++filt", tokens[5] ]).decode("utf-8")
      sym = sym.replace('\n', '')
      sym = SaneBase64(sym)
      result[address] = sym
  return result

def find_nth(haystack, needle, n):
  start = haystack.find(needle)
  while start >= 0 and n > 1:
    start = haystack.find(needle, start+len(needle))
    n -= 1
  return start

def ObtainPEFunctionSymbols(training_file):
  result = {}
  filetype = subprocess.check_output(["file", "-b", training_file]).decode("utf-8")
  if filetype == "PE32+ executable (console) x86-64, for MS Windows\n":
    default_base = 0x140000000
  elif filetype == "PE32+ executable (DLL) (console) x86-64, for MS Windows\n":
    default_base = 0x180000000
  elif filetype == "PE32 executable (console) Intel 80386, for MS Windows\n":
    default_base = 0x400000
  elif filetype == "PE32 executable (DLL) (GUI) Intel 80386, for MS Windows\n":
    default_base = 0x10000000
  elif filetype == "PE32 executable (DLL) (console) Intel 80386, for MS Windows\n":
    default_base = 0x10000000
  else:
    print("Problem: %s has unknown file type" % training_file)
    print("Filetype is: %s" % filetype)
    sys.exit(1)
  if not os.path.isfile(training_file + ".debugdump"):
    print("No .debugdump file found, no debug symbols...", end ='')
    return result
  try:
    function_lines = [
      line for line in open(training_file + ".debugdump", "rt", errors='ignore').readlines() if
      line.find("Function") != -1 and line.find("static") != -1
      and line.find("crt") == -1 ]
  except:
    # No debugdump data found
    print("Failed to load debug data.")
    traceback.print_exc(file=sys.stdout)
    return result
  for line in function_lines:
    # The lines we wish to split are of the form:
    # Function : static, [
    symbol = line[ find_nth(line, ", ", 3) + 2 :]
    if line.find("[") == -1:
      continue
    try:
      address = int(line.split("[")[1].split("]")[0], 16) + default_base
    except:
      print("Invalid line, failed to split - %s" % line)
      traceback.print_exc(file=sys.stdout)
      continue
    # We still need to stem and encode the symbol.
    stemmed_symbol = subprocess.run(["../bin/stemsymbol"], stdout=PIPE,
      input=bytes(symbol, encoding="utf-8")).stdout
    if len(stemmed_symbol) > 0:
      result[address] = SaneBase64(stemmed_symbol.decode("utf-8"))
  return result

def ObtainDisassembledFunctions(training_file_id):
  """ Returns a sorted list with the functions that the disassembler found.
  """
  try:
    inputdata = open( FLAGS.work_directory + "/" + "functions_%s.txt" % training_file_id,
      "rt" ).readlines()
  except:
    print("Could not open functions_%s.txt, returning empty list." % training_file_id)
    return []
  data = []
  try:
    # This used to be a single-line list comprehension, but had to be rewritten
    # to make sure we are not throwing exceptions.
    # data = [ int(line.split()[0].split(":")[1], 16) for line in inputdata ]
    for line in inputdata:
      token = line.split()
      if len(token) >= 1:
        token2 = token[0].split(":")
        if len(token2) >= 2:
          data.append(int(token2[1], 16))
  except:
    # Bizarre, this should not happen?
    print("Exception when parsing functions_%s.txt?" % training_file_id)
    print("Line that caused the exception is %s" % line)
    traceback.print_exc(file=sys.stdout)
  data.sort()
  return data

def RunJSONDotgraphs(argument_tuple):
  """ Run the bin/dotgraphs utility to generate JSON output from the
  disassembly and write the output to a directory named after the file id. """
  training_file = argument_tuple[0]
  file_id = argument_tuple[1]
  file_format = argument_tuple[2]

  # Make the directory.
  directory_name = FLAGS.work_directory + "/" + "json_%s" % file_id
  shutil.rmtree(directory_name, ignore_errors=True)
  try:
    os.mkdir(directory_name)
  except OSError:
    print("Directory %s exists already." % directory_name)

  try:
    fingerprints = subprocess.check_call(
      [ "../bin/dotgraphs", "--no_shared_blocks", "--json",
      "--format=%s" % file_format, "--input=%s" % training_file, 
      "--output=%s" % directory_name])
  except:
    print("Failure to run dotgraphs (%s:%s->%s)" % \
      (file_format, training_file, file_id))

  print("Done with dotgraphs. (%s:%s->%s)" % \
    (file_format, training_file, file_id))

def RunFunctionFingerprints(argument_tuple):
  """ Run the bin/functionfingerprints utility to generate features from the
  disassembly and write the output to a file named after the file id. """
  training_file = argument_tuple[0]
  file_id = argument_tuple[1]
  file_format = argument_tuple[2]
  if FLAGS.disable_mnemonic:
    mnemonic = "--disable_instructions=true"
  else:
    mnemonic = "--disable_instructions=false"
 
  write_fingerprints = open(
    FLAGS.work_directory + "/" + "functions_%s.txt" % file_id, "wt")
  
  try:
    fingerprints = subprocess.check_call(
      [ "../bin/functionfingerprints", "--no_shared_blocks",
      "--format=%s" % file_format, "--input=%s" % training_file, 
      "--minimum_function_size=5", "--verbose=true", mnemonic ], 
      stdout = write_fingerprints)
  except:
    print("Failure to run functionfingerprints (%s:%s->%s)" % \
      (file_format, training_file, file_id))

  write_fingerprints.close()
  print("Done with functionfingerprints. (%s:%s->%s)" % \
    (file_format, training_file, file_id))

def ProcessTrainingFiles(training_files, file_format):
  # Begin by launching a pool of parallel processes to call
  # RunFunctionFingerprints.
  argument_tuples = []
  for training_file in training_files:
    sha256sum = subprocess.check_output(["sha256sum", training_file]).split()[0]
    file_id = sha256sum[0:16].decode("utf-8")
    argument_tuples.append((training_file, file_id, file_format))
  # Use a quarter of the available cores.
  pool = multiprocessing.Pool(max(1, int(FLAGS.parallelism)))
  if FLAGS.generate_fingerprints:
    print("Running functionfingerprints on all files.")
    pool.map(RunFunctionFingerprints, argument_tuples)
  if FLAGS.generate_json_data:
    print("Running dotgraphs on all files.")
    pool.map(RunJSONDotgraphs, argument_tuples)

  for training_file in training_files:
    sha256sum = subprocess.check_output(["sha256sum", training_file]).split()[0]
    file_id = sha256sum[0:16].decode("utf-8")
    # Run objdump
    print("Obtaining function symbols from %s... " % training_file, end='')
    objdump_symbols = ObtainFunctionSymbols(training_file, file_format)
    print("got %d symbols..." % len(objdump_symbols), end='')
    # Get the functions that our disassembly could find.
    print("Getting disassembled functions. ", end='')
    disassembled_functions = ObtainDisassembledFunctions(file_id)
    print("got %d functions..." % len(disassembled_functions), end='')
    # Write the symbols for the functions that the disassembly found.
    if len(objdump_symbols) > 0:
      print("Opening and writing extracted_symbols_%s.txt. " % file_id, end='')
      output_file = open( FLAGS.work_directory + "/" +
        "extracted_symbols_%s.txt" % file_id, "wt" )
      symbols_to_write = []
      for function_address in disassembled_functions:
        if function_address in objdump_symbols:
          symbols_to_write.append((function_address,
            objdump_symbols[function_address]))
      print("Sorting...", end='')
      symbols_to_write.sort(key=lambda a: a[1].lower())
      # symbols_to_write contains only those functions that are both in the dis-
      # assembly and that we have symbols for.
      print("Writing...", end='')
      count = 0
      for address, symbol in symbols_to_write:
        output_string = "%s %s %16.16lx %s false\n" % (file_id, training_file,
          address, symbol)
        output_file.write(output_string)
        count = count + 1
      print("Done (wrote %d symbols)" % count)
      output_file.close()
    else:
      print("No symbols. Skipping.")

def BuildSymbolToFileAddressMapping():
  """
  Constructs a map of symbol-string -> [ (file_id, address), ... ] so that each
  symbol is associated with all the files and addresses where it occurs.
  """
  result = defaultdict(list)
  # Iterate over all the extracted_symbols_*.txt files.
  for filename in os.listdir(FLAGS.work_directory):
    print("Checking filename %s" % filename)
    if fnmatch.fnmatch(filename, "extracted_symbols_*.txt"):
      print("Processing file %s" % filename)
      contents = open( FLAGS.work_directory + "/" + filename, "rt" ).readlines()
      for line in contents:
        file_id, filename, address, symbol, vuln = line.split()
        result[symbol].append((file_id, address))
  return result

def IndexToRowColumn(index, n):
  """
    Given an index into the non-zero elements of an upper triangular matrix,
    returns a tuple of integers indicating the row, column of that entry.

    n is the number of elements in the family we are dealing with.
  """
  if n & 1:
    # Code for uneven n. Illustration using n=5.
    #
    # 01234      A1234
    # 00567      98567
    # 00089  =>
    # 0000A
    # 00000
    #
    row_index = index / n
    column_index = index % n
    if column_index <= row_index:
      row_index = n - row_index - 2
      column_index = n - column_index - 1
    else:
      row_index = row_index
      column_index = column_index
    return (row_index, column_index)
  else:
    # Code for even n. Illustration using n=6. The matrix we care about looks
    # like this, but can be transformed into a square with n=5.
    #
    # 012345    12345
    # 006789    F6789
    # 000ABC => EDABC
    # 0000DE
    # 00000F
    # 000000
    #
    row_index = index / (n-1)
    column_index = index % (n-1)
    if column_index < row_index:
      # Deal with the mirroring. Our row index is now the index from the end.
      row_index = n - row_index - 1
      column_index = n - column_index - 1
    else:
      row_index = row_index
      column_index = column_index + 1
    return (row_index, column_index)

def FamilySize(n):
  return (n**2 - n) / 2

def WriteAttractAndRepulseFromMap( input_map, output_directory,
  number_of_pairs=1000, return_sets=False ):
  """
  Creates attraction and repulsion pairs (and writes attract.txt and repulse.txt
  to the output_directory, for the case of "generalization performance to unseen
  samples".
  """
  # Begin by calculating the total number of possible attraction pairs.
  symbol_to_index = []
  total_number_of_attraction_pairs = 0
  for symbol, file_and_address in input_map.items():
    n = len(file_and_address)
    symbol_to_index.append((total_number_of_attraction_pairs, symbol))
    total_number_of_attraction_pairs = total_number_of_attraction_pairs +\
      FamilySize(n)
  # Choose a random subset of number_of_pairs size of these pairs. We choose
  # indices first, and then generate the pairs thereafter.
  indices = set()
  print("Attraction: Requested %d pairs with %d available." % (number_of_pairs,
    int(total_number_of_attraction_pairs)))
  if (total_number_of_attraction_pairs > 0x7FFFFFFFFFFFFFFF):
    # We cannot use numpy.choice on numbers that do not fit into int64, so
    # generate a list of regular integers of the sufficient size
    # probabilistically.
    while len(indices) != number_of_pairs:
      # From a CS-perspective, this could loop forever, but should not in
      # practice.
      indices.add(random.randrange(total_number_of_attraction_pairs))
  elif number_of_pairs < total_number_of_attraction_pairs:
    # Request is for fewer pairs than are available.
    indices = set(numpy.random.choice(int(total_number_of_attraction_pairs),
      number_of_pairs, replace=False))
  else:
    # Request is for the maximum number of pairs available.
    indices = set(range(int(total_number_of_attraction_pairs)))
  # We should have a set of indices for the attract.txt pairs. Now generate
  # the actual pairs.
  attraction_set = set()
  for index in indices:
    # First find the symbol into whose family the index falls.
    family_index = max(bisect.bisect(symbol_to_index, (index, 'a')) - 1, 0)
    family_start, family_symbol = symbol_to_index[family_index]
    n = len(input_map[family_symbol])
    family_size = FamilySize(n)
    within_family_index = index - family_start
    family_size = len(input_map[family_symbol])
    row, column = IndexToRowColumn(within_family_index, n)
    row = int(row)
    column = int(column)
    attraction_set.add((input_map[family_symbol][row],
      input_map[family_symbol][column]))
  # The next step is generating repulsion pairs.
  repulsion_set = GenerateRepulsionPairs( input_map, number_of_pairs )
  if return_sets:
    return (attraction_set, repulsion_set)
  # Write the files.
  WritePairsFile( attraction_set, output_directory + "/attract.txt" )
  WritePairsFile( repulsion_set, output_directory + "/repulse.txt" )
  return

def GenerateRepulsionPairs( input_map, number_of_pairs ):
  repulsion_set = set()
  max_loop_iterations = number_of_pairs**3
  symbols_as_list = list(input_map.keys())
  symbols_as_list.sort()
  while len(repulsion_set) != number_of_pairs and max_loop_iterations > 0:
    symbol_one, symbol_two = numpy.random.choice( symbols_as_list, 2,
      replace=False )
    element_one = random.choice( input_map[symbol_one] )
    element_two = random.choice( input_map[symbol_two] )
    ordered_pair = tuple(sorted([element_one, element_two]))
    repulsion_set.add(ordered_pair)
    max_loop_iterations = max_loop_iterations - 1
  return repulsion_set

def WritePairsFile( set_of_pairs, output_name ):
  """
  Take a set of pairs ((file_idA, addressA), (file_idB, addressB)) and write them
  into a file as:
    file_idA:addressA file_idB:addressB
  """
  result = open(output_name, "wt")
  for pair in set_of_pairs:
    result.write("%s:%s %s:%s\n" % (pair[0][0], pair[0][1], pair[1][0],
      pair[1][1]))
  result.close()

def WriteFunctionsTxt( output_directory ):
  """
  Simply copy all the contents of all the functions_*.txt files into the output
  directory.
  """
  output_file = open( output_directory + "/functions.txt", "wt" )
  for filename in os.listdir(FLAGS.work_directory):
    if fnmatch.fnmatch(filename, "functions_????????????????.txt"):
      data = open(FLAGS.work_directory + "/" + filename, "rt").read()
      output_file.write(data)
  output_file.close()





def SplitFamilies(symbol_dict, percentage_list):
  result = [defaultdict(list) for _ in percentage_list]
  for key, value in symbol_dict.items():
    idx = np.random.choice(len(percentage_list), p=percentage_list)
    result[idx][key] = value
  return result

def GenerateAllPairs(symbol_dict, return_sets, output_directory=""):
  if not return_sets:
    assert(not output_directory=="")
  return WriteAttractAndRepulseFromMap( symbol_dict,
    output_directory, number_of_pairs=100, return_sets=return_sets)  

def SplitGraphs(symbol_dict, percentage_list):
  # Need to ensure that there is at least one graph of each family in training
  num_splits = percentage_list
  result = [defaultdict(list) for _ in percentage_list]
  for function_family, elements in symbol_to_file_and_address.items():
    if len(elements) < num_splits:
      continue
    for idx in range(num_splits):
      elem = elements.pop()
      result[idx][function_family].append(elem)
    for elem in elements:
      idx = np.random.choice(num_splits, p=percentage_list)
      result[idx][function_family].append(elem)
  return result

def SplitPairs(symbol_dict, percentage_list):

  (attraction_set, repulsion_set) = GenerateAllPairs(symbol_dict, return_sets=True)
  a_l, r_l = list(attraction_set), list(repulsion_set)
  a = len(attraction_list)
  r = len(repulsion_list)
  p = np.cumsum(percentage_list)
  assert(p[2] == 1)
  attract_train, attract_val, attract_test = a_l[0:p[0]*a], a_l[p[0]*a:p[1]*a], a_l[p[1]*a:a]
  repulse_train, repulse_val, repulse_test = r_l[0:p[0]*r], r_l[p[0]*r:p[1]*r], r_l[p[1]*r:r]

  train = attract_train + repulse_train
  val = attract_val + repulse_val
  test = attract_test + repulse_test

  # Make sure every graph is in the training set
  print("Start checking for graphs missing in training set")

  train_graphs = set()
  for pair in train:
    train_graphs.update([pair[0], pair[1])

  move_list = []
  for idx, pair in enumerate(val):
    if (not pair[0] in train_graphs) or (not pair[1] in train_graphs):
      train_graphs.update([pair[0], pair[1])
      move_list.append(idx)

  for idx in reversed(move_list)
    train.append(val.pop(odx))

  move_list = []
  for idx, pair in enumerate(test):
    if (not pair[0] in train_graphs) or (not pair[1] in train_graphs):
      train_graphs.update([pair[0], pair[1])
      move_list.append(idx)

  for idx in reversed(move_list)
    train.append(val.pop(odx))

  total = len(train) + len(val) + len(test)
  print(f"Final Split is: {len(train)/total}/{len(val)/total}/{len(test)}")

  return train, val, test





def main(argv):
  del argv # unused.

  # Refuse to run on Python less than 3.5 (unpredictable!).

  if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    print("This script requires Python version 3.5 or higher.")
    sys.exit(1)

  if FLAGS.clobber:
    shutil.rmtree(FLAGS.work_directory)
    os.mkdir(FLAGS.work_directory)

  if FLAGS.work_directory[-1] != '/':
    FLAGS.work_directory = FLAGS.work_directory + '/'

  print("Processing ELF training files to extract features...")
  ProcessTrainingFiles(FindELFTrainingFiles(), "ELF")
  print("Processing PE training files to extract features...")
  ProcessTrainingFiles(FindPETrainingFiles(), "PE")

  # We now have the extracted symbols in a set of files called
  # "extracted_symbols_*.txt"

  # Get a map that maps every symbol to the files in which it occurs.
  print("Loading all extracted symbols and grouping them...")
  symbol_to_files_and_address = BuildSymbolToFileAddressMapping()

  print(len(symbol_to_files_and_address))

  for i, (symbol, file_and_address) in enumerate(symbol_to_files_and_address.items()):
    print(i)
    print(symbol)
    print(file_and_address)
    if i > 5:
      break
  
  np.random.seed(42)
  random.seed(42)

  # split function-families
  chunk, val3, test3 = SplitFamilies(symbol_to_files_and_address, [.8, .1, .1])
  GenerateAllPairs(val3, return_sets=False, 
    output_directory=FLAGS.work_directory + "/val3",)
  GenerateAllPairs(test3, return_sets=False, 
    output_directory=FLAGS.work_directory + "/test3")
    
  # simple alternative training set (no fancy validation)
  GenerateAllPairs(chunk, return_sets=False, 
    output_directory=FLAGS.work_directory + "/train_all")

  # split function-graphs
  chunk, val2, test2 = SplitGraphs(chunk, [.8, .1, .1])
  # could instead do pairs across like the original code
  GenerateAllPairs(val2, return_sets=False, 
    output_directory=FLAGS.work_directory + "/val2")
  GenerateAllPairs(test2, return_sets=False, 
    output_directory=FLAGS.work_directory + "/test2")

  # split graph-pairs
  train, val1, test1 = SplitPairs(chunk, [.8, .1, .1])

  WritePairsFile( train[0], output_directory + "/train/attract.txt" )
  WritePairsFile( train[1], output_directory + "/train/repulse.txt" )
  WritePairsFile( val1[0], output_directory + "/val1/attract.txt" )
  WritePairsFile( val1[1], output_directory + "/val1/repulse.txt" )
  WritePairsFile( test1[0], output_directory + "/test1/attract.txt" )
  WritePairsFile( test1[1], output_directory + "/test1/repulse.txt" )

  print("Done, ready to run training.")

if __name__ == '__main__':
  app.run(main)

