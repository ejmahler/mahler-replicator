# =================================================
# A Golly script to write the program and data
# tapes for the Devore-Binary UCC. Load devore-binary.rle
# and run this script, then set the machine running.
#
# Tape length: 15,513
# Rep period: 628,311,357 timesteps
#
# Script license: public domain
#
# Contact: 
# Elliott Mahler jointogether (at) gmai1 (dot) com
#
# This project wouldn't exist without the foundations provided by the devore-rep machine.
# To learn more about the design and history of this replicator, devore-body.rle and make-Devore-tape.py would be a good place to start
#
# Thanks to Tim Hutton, Ron Hightower, and John Devore
# =================================================
#
#
# "Mahler-Replicator" is a classical Universal Turing Machine whose main purpose is to create copies of itself. It is run with the Devore2 ruleset.
#
# Its main features are:
# - A data arm, with a read/write data head that can serve to interact with the machine's "main memory", represented as a long string of bits called the "data tape".
# - A program arm, with a read-only head that can read instructions from an encoded computer program known as the "program tape".
# - A "construction arm", which can be maneuvered to read and write arbitrary cells.
#
# This file contains documentation for this machine, a compiler that creates programs for this machine, and a sample program for the compiler to compile.








'''
-- MACHINE OVERVIEW --

This section contains a high-level overview of the machine, including a list of al lthe instructions it can execute, documentation of machine features, etc.

INTERNAL STATE
This machine has two internal registers:
- 	R: the R register stores the results of read operations (for example, when the machine reads the current value of the data tape). R has two possible values: 0 or 1
		The jump_if_zero and jump_if_one instructions check the current value of R and jump (or don't jump) accordingly.
		R has an initial value of 1.
		Instructions prefixed with "R_" will overwrite the current value of R.
- 	A: the A register is a 5-bit incrementing register. A starts at 0, and can be incremented up to a value of 31, after which another increment will roll it over to a value of 0.
		After every increment, the machine will store 1 in R if the new value of A is 0, otherwise it stores 0 in R
		
		A can be used to implement fixed-sized loops. To do this, write a label at the beginning of your loop, increment inside the loop, and after the increment, jump_if_zero back to the beginning of the loop.
		A can also be used to check whether some unknown quantity [such as the number of times we've extended while copying the data tape] is a multiple of 32 or not.
		Instructions prefixed with "A_" will overwrite the current value of A.


INSTRUCTIONS
Each instruction is represented by 4 binary bits, for a total of 16 instructions.
'''					

jump_if_one = '0000'				# Jump if R==1 (Following bits give distance to jump)
jump_if_zero = '0001'				# Jump if R==0 (Following bits give distance to jump)
R_data_head_left = '0010'			# Move data head left, read, and store the result in R
R_data_head_right = '0011'			# Move data head right, read, and store the result in R
toggle_C_or_D = '0100'				# Switch between construction path and data path [D open by default]
R_read = '0101'						# Read the current value under the construction arm or data arm, and store the result in R [affects whichever of C or D is open]
mark = '0110'						# Set the cell under the construction arm or data arm to 1 [affects whichever of C or D is open]
erase = '0111'						# Set the cell under the construction arm or data arm to 0 [affects whichever of C or D is open]
AR_extend_and_increment_2x = '1000'	# [Same thing as writing 2 consecutive "AR_extend_and_increment" instructions]
AR_extend_and_increment = '1001'	# Increment A, then read its new value and store the result in R. Simultaneously, execute the "extend" instruction

# these next instructions maneuver the construction arm
extend_2x = '1010'					# [Same thing as writing 2 consecutive "extend" instructions]
extend = '1011'						#
extend_left = '1100'				#
extend_right = '1101'				#
retract_16x = '1110'				# [Same thing as writing 16 consecutive 'retract' instructions]
retract = '1111'					# [this instruction also handles retract_left and retract_right]

# we also have several macros which are combinations of other instructions
MACRO_R_readleft = extend_left + extend + R_read + retract + retract
MACRO_markleft = extend_left + extend + mark + retract + retract
MACRO_eraseleft = extend_left + extend + erase + retract + retract
MACRO_printright = extend_right + mark + retract
MACRO_AR_increment_A = AR_extend_and_increment + retract # It's so rare to need to increment A without extending that it's not worth having a dedicated instruction for it.


'''
WRITING JUMPS/BRANCHES
Jump destinations are called 'labels'. They're specified in the program by writing a string just before the instruction to jump to, with the format '@label_name'.
Labels are free: The machine itself doesn't use labels in any way. The compiler uses them to compute jump addresses, and prints them underneath the program tape for debugging purposes, but does not encode them into the program tape or data tape.

When writing a jump, instead of directly writing the jump instruction, write a string with the following format: 'jump_if_one:destination_label_name' or 'jump_if_zero:destination_label_name'

The maximimum jump distance is 127 words (An instruction is 1 word, and a jump address is 2 words) forward or backward from the end of the jump address. The compiler will raise an error if you try to jump farther than that.





GOTO
This machine does not have a goto instruction. If you write a goto in the program, the compiler will try to prove what R currently contains.
- The machine starts out with R=1, so R is guaranteed to be one until something sets it otherwise.
- Whenever we see a label, or an instruction prefixed with R_, the current value of R becomes unknown.
- If we see a jump_if_one, thenwe know R=0 immediately afterwards (Because if R was one, the jump would have been taken), and vice versa for jump_if_zero

If the compiler can prove that R=1, it will replace the goto wilth a jump_if_one instruction. if it can prove that R=0, it will replace the goto with a jump_if_zero
If it can't prove either, it will insert a jump_if_zero instruction followed by a jump_if_one. In practice, almost all gotos immediately follow a jump_if_zero or jump_if_one instruction, so this insert is rarely needed.




INJECTION (for bootstrapping a copy of the machine)
For injection, use the pattern:
1 1
  1 1 1
1 1
		 
Approach from the left center until the construction head is inside the reciever, then retract. This begins the sheathing process.
Then retract again to inject the initializing signal. You can then safely retract away. From the perspective of the machine, the injection appears as if the machine has just read a 1 from the construction arm.
It will store a 1 in R, and then begin executing instructions just like it does after any other C or D read.



RETRACT
Retract, retract_left, and retract_right are no longer distinct instructions, the single "retract" instruction can handle all three without any special treatment.

One caveat is that you must take care to avoid zigzags in the construction arm, which I call "kinks". For example, if you extend left, then immediately extend right,
you've created a kink in the construction arm, and the construction arm will break if you try to retract out of the kink.



-- COMPILER --
This section contains the implementation of the compiler.
'''

from collections import namedtuple, OrderedDict

# create a generator that yields N-sized chunks of the provided collection
def chunks(collection, chunk_size):
	for i in xrange(0, len(collection), chunk_size):
		yield collection[i:i+chunk_size]

# encodes the given value into a string of num_bits of binary, MSB first
# if num_bits is too small to represent the given value, returns a string of all 1's
def encode_binary(value, num_bits):
	result = ''
	while num_bits > 0:
		num_bits -= 1

		digit = 1 << num_bits

		if value >= digit:
			value -= digit
			result += '1'
		else:
			result += '0'
	return result

# Compiler utilities
Branch = namedtuple('Branch', ['line_number', 'line_text', 'target_label', 'origin_address', 'address_index'])
class Label(object):
	def __init__(self, line_number, name, address):
		self.line_number = line_number
		self.name = name
		self.address = address
		self.num_references = 0

	def add_reference(self):
		self.num_references += 1

INSTRUCTION_SIZE = 4
JUMP_ADDRESS_SIZE = 8

def compile_program(program):
	'''Takes the given program, represented as a List of instructions, and returns a string of encoded binary comprising the compiled program'''

	# set of branch statements. whenever we encounter an instruction contained in this set, we know we have to do the extra work associated with a branch
	branch_statements = {
		'goto',
		'jump_if_one',
		'jump_if_zero',
	}

	# these statements overwrite R with a value that won't be known until runtime
	read_statements = {
		R_read,
		R_data_head_left,
		R_data_head_right,
		AR_extend_and_increment,
		AR_extend_and_increment_2x
	}

	# store a list of encoded instructions for our program, ie store the actual binary for each instruction
	encoded_program = []

	# store a dict from labels to their indexes
	labels = OrderedDict()

	# likewise, store a list of branches and the indexes we found them at
	branches = []

	# at certain points in the program, we 100% know that R contains 0 or 1. for example, at the start of the program R will always contain 1
	current_R = 1

	# the compiler runs in two phases
	# this loop is the first phase (compile phase), whre we encode each instruction into binar,y and record where we foundbranches and labels
	current_address = 0
	for i, line in enumerate(program):

		if line.startswith('@'):
			# This is a label, and anything could be jumping to this, so we no longer know what R contains
			#we could look backwards through 
			current_R = None

			# record the label and its address
			label_name = line[1:]
			labels[label_name] = Label(i, label_name, current_address)
			continue

		# this is an actual instruction, decode it here
		instruction, sep, target = line.partition(':')
		
		# if this is a branch statement, it's probably specified in text form, rather than in binary form. figure out what actual instruction to use, based on the text
		if instruction in branch_statements:
			if target is None:
				g.exit("Invalid program: The jump on line %d must be accompanied by a label, in the format 'jump_if_x:label_name'"%i)

			#if we've already seen the target label, this is a backwards jump
			backwards = target in labels

			#if this branch is jump_if_one or jump_if_zero, we know that R is guaranteed to contain the opposite value if the jump isn't taken
			if instruction == 'jump_if_one':
				instruction = jump_if_one
				current_R = 0

			elif instruction == 'jump_if_zero':
				instruction = jump_if_zero
				current_R = 1

			#if this branch is a goto, we have to do some rewriting, because goto doesn't actually exist.
			#if we can prove that R is 1 or 0, jump on that value. otherwise, insert an instruction to set R to 0, followed by a jump_if_zero
			elif instruction == 'goto':
				if current_R == 1:
					instruction = jump_if_one
				elif current_R == 0:
					instruction = jump_if_zero
				else:
					# this is a goto
					# we can't prove what R contains, so we have to insert a jump_if_zero, and follow it with a jump_if_one
					instruction = jump_if_zero
					encoded_program.append(jump_if_one)

					current_address += INSTRUCTION_SIZE
					current_address += JUMP_ADDRESS_SIZE

					# record some important data about this branch so that we can come back later and fix the placeholder
					branches.append(Branch(i, line, target, current_address, len(encoded_program)))

					# add a placeholder for the address offset
					encoded_program.append('')

		#if this is a valid instruction, add it it the decoded program list
		#note that we only check that the length is dividible by 4, not that it's exactly 4. this is because this "instruction" could be a macro
		if len(instruction)%INSTRUCTION_SIZE == 0 and all(bit == '0' or bit == '1' for bit in instruction):
			current_address += len(instruction)
			encoded_program.append(instruction)

			for chunk in chunks(instruction, INSTRUCTION_SIZE):
				if chunk in read_statements:
					# since we're reading runtime data into R, we no longer know what it contains
					current_R = None
		else:
			g.exit("Invalid program: Unrecognized instruction on line %d: %s"%(i, line))

		# if there is a target, this is a branch
		if target:

			# this is a branch instruction, keep track of the size of the address
			current_address += JUMP_ADDRESS_SIZE

			# record some important data about this branch so that we can come back later and fix the placeholder
			branches.append(Branch(i, line, target, current_address, len(encoded_program)))

			# add a placeholder for the address offset
			encoded_program.append('')

	# second compile phase (aka the linker phase)
	# now that we know all the labels and their locations, go through all the branches and fix up their destination addresses
	for branch in branches:
		if branch.target_label not in labels:
			g.exit("Invalid program: A jump statement on line %d referenced a label that doesn't exist. Line text: %s"%(branch.line_number, branch.line_text))

		target = labels[branch.target_label]

		#Record the fact that this label was referenced
		target.add_reference()

		offset = (target.address - branch.origin_address)

		# write the absolute value of the offset in binary, MSB first
		serialized_offset = abs(offset) / 4

		max_offset = 127
		if serialized_offset > max_offset:
			g.exit("Invalid program: The jump statement '%s' referenced a label that is too far away. (The distance is %d, the maximum is %d words)"%(branch.line_text, offset, max_offset))

		direction_string = "1" if offset > 0 else "0"
		address_string = encode_binary(serialized_offset, 7)
		encoded_program[branch.address_index] = direction_string + address_string

	# convert the program data to a single binary string
	program_string = ''.join(encoded_program)
	label_data = list(labels.itervalues())

	return (program_string, label_data)




# -- PROGRAM --
# This section contains the self-replication program. The "program" array contains a list of labels and instructions that will be encoded into binary by the compiler.
#
# Note that the farther the program arm is extended, the more cycles it takes to complete the round-trip to read the tape.
# Because of this, instructions and subroutines are reordered to ensure that the most-used sections of code are closest to the machine
# There's a balance to strike here between "most-used instructions first" and "too many jumps making the program tape longer".

program = [
	#initialize the machine
	'goto:initialize_bridge',

	# ---------- copy phase: ------------------------
	# == Copy a 1 from the data tape ==
	'@copyphase',
	AR_extend_and_increment,
	'@copyphase_pre_extended', # We implement an opimization down below where we extend 2x instead of once whenever possible. If we extend 2x then we can skip "normal" extend by jumping to this label instead of to "copyphase"
	MACRO_printright,
	R_data_head_right,
	'jump_if_one:copyphase',
	# == 0 ==
	AR_extend_and_increment_2x,
	R_data_head_right,
	'jump_if_one:copyphase_pre_extended',
	# == 00 ==
	R_data_head_right,
	'jump_if_one:copyphase',
	# == 000 ==
	AR_extend_and_increment_2x,
	R_data_head_right,
	'jump_if_one:copyphase_pre_extended',
	# == 0000 ==
	R_data_head_right,
	'jump_if_one:copyphase',
	# == 00000 ==
	AR_extend_and_increment_2x,
	R_data_head_right,
	'jump_if_one:copyphase_pre_extended',
	# == 000000 ==
	R_data_head_right,
	'jump_if_one:copyphase',
	# == 0000000 ==
	# -- end of copy --
	'goto:copyphase_finalize',

	# ---------- construction phase: ------------------------  

	# Write ones in a loop until we see a 0 i nthe data tape
	'@1',
	AR_extend_and_increment,
	MACRO_printright,
	R_data_head_left,
	'jump_if_one:1',

	#tape has a 0, so we're going to write a string of zeroes
	'@zero_block',
	R_data_head_left,
	'jump_if_zero:zero_block_evens',

	# this chunk has an odd number of zeroes -- handle the odd one now and then fall through to the evens code
	AR_extend_and_increment,
	R_data_head_left,
	'jump_if_one:1',

	#from here on out we're going to assume that we're writing an even-sized string of zeroes
	'@zero_block_evens',
	AR_extend_and_increment_2x, #after this instruction, we've written either 2 or 3 zeroes
	R_data_head_left,
	'jump_if_one:1',

	AR_extend_and_increment_2x, #after this instruction, we've written either 4 or 5 zeroes
	R_data_head_left,
	'jump_if_one:1',

	AR_extend_and_increment_2x, #after this instruction, we've written either 6 or 7 zeroes
	R_data_head_left,
	'jump_if_one:1',

	# at this point, we're either writing a repeated sequence of chunks or we're ending the row
	R_data_head_left,
	'jump_if_zero:construction_endrow',

	# we're writing a sequence of chunks. first we have to check if we're already done, because we consumed some of the chunk as we decoded
	R_data_head_left,
	'jump_if_zero:zero_block_evens',

	'@zero_chunk',
	AR_extend_and_increment_2x,AR_extend_and_increment_2x,AR_extend_and_increment_2x,
	R_data_head_left,
	'jump_if_one:zero_chunk',
	'goto:zero_block_evens',

	# ---------- jump bridges  ------------------------
	# some jumps have too long of an offset, we use these as halfway points to reduce the jump distance

	# we know for an absolute fact that R will be one in both of these cases -- but the compiler doesn't, because it can't follow jumps. when it sees a label, it discards all information it knows about the contents of R
	# to avoid having a jump_if_zero and jump_if_one here, we'll just hardcode jump_if_one. TODO: improve the compiler to autodetect this
	'@copyphase_bridge',
	'jump_if_one:copyphase_pre_extended',

	'@initialize_bridge',
	'jump_if_one:initialize',

	# ---------- construction phase (more): ------------------------

	'@construction_endrow',

	# in "align to A" below, we're going to extend past the end of the machine - but we might hit the data tape! to prevent that, retract twice first
	retract_16x,
	retract_16x,

	# extend and increment until A is zero
	'@align_to_a',
	AR_extend_and_increment,
	'jump_if_zero:align_to_a',

	# now that we know we've extended a multiple of 32 times, we can retract 32 cells per read, significantly reducing the number of expensive reads we have to execute
	'@retractrow',
	retract_16x,
	retract_16x,
	MACRO_R_readleft,
	'jump_if_zero:retractrow',
	MACRO_eraseleft,

	# next bit indicates whether we have another row. 1=yes, 0=no
	R_data_head_left,
	'jump_if_zero:finish',

	# begin constructing the next row
	extend,

	# make a mark at the start of the row so we know where to stop when retracting
	MACRO_markleft,

	# move into writing position, keeping track of how many times we extend or extend right while doing so
	'@constructionphase', # when jumping from copy phase finalization to the start of the construction phase, this will be our starting point
	extend_right,
	MACRO_AR_increment_A, #increment once to account for the extend right we just did
	AR_extend_and_increment_2x,

	# begin construction
	R_data_head_left,
	'jump_if_one:1',
	'goto:zero_block',

	

	# ---------- copy phase (finalization): ------------------------

	# the copy phase has ended. next we have to construction arm to a multiple of 32
	'@copyphase_finalize',
	AR_extend_and_increment,
	'jump_if_zero:copyphase_finalize',

	# We know through magic that the construction arm is actually 5 short of where we think it is. If thie machine size changes, this will need to change here!
	# Ideally this would be handled in the @initialize section, but for this very specific machine size it's simplest to deal with it here
	extend_2x,extend_2x,extend,

	# now that we know we've extended a multiple of 32 times, we can retract 32 cells per read, significantly reducing the number of expensive reads we have to execute
	# The machine will spend around 10% of its time executing the following loop. (it would be much more if we didn't have retract_16x and the A register!)
	'@retractcopy',
	retract_16x,
	retract_16x,
	MACRO_R_readleft,
	'jump_if_zero:retractcopy',

	# we overshot on the data tape while copying, move back into position to begin construction
	'@find_data_start',
	R_data_head_left,
	'jump_if_zero:find_data_start',

	# we are now ready to begin construction
	'goto:constructionphase',





	# ------------------------ subroutines (here for speed): -----------------

	'@initialize',
	# move the construction arm into the starting position for copying the data tape

	# extend left to make clearance for the new machine
	# note that we're incrementing the A register a total of 10 times here before we enter @extendup_loop, so that it only iterates 22 times instead of 32
	AR_extend_and_increment_2x,AR_extend_and_increment_2x,AR_extend_and_increment,

	extend_right,

	# the number of times we extend here determines the vertical separation between the top of the current machine and the bottom of the new one.
	# When we start construction, the place we stop here will be our starting height for constructing the first row
	AR_extend_and_increment_2x, AR_extend_and_increment_2x,AR_extend_and_increment_2x,extend,

	#make a mark so that we know where to stop when retracting
	toggle_C_or_D, #open C
	MACRO_markleft,

	# extend upwards 110 times
	# Because we incremented 10 times up above while moving into position, this loop will run 22 times. we extend 5 times inside the loop, 22 * 5 = 110
	'@extendup_loop',
	AR_extend_and_increment,extend_2x,extend_2x,
	'jump_if_zero:extendup_loop',
	extend,

	# we've extended as high as we need to go
	extend_right,

	# extend to the right 144 times
	'@extendright_loop',
	AR_extend_and_increment_2x,extend_2x,extend_2x,extend_2x,extend,
	'jump_if_zero:extendright_loop',

	# move the data head and copy arm right until we hit the start of the data tape
	'@find_data_tape',
	AR_extend_and_increment,
	R_data_head_right,
	'jump_if_zero:find_data_tape',
	'goto:copyphase_bridge',



	#Finishing phase: end the construction phase and inject the next machine
	'@finish',

	# Our goal is to extend into the injection reciever, then retract to perform the injection. We've arranged things so that the injection reciever is only a few cells away right now
	retract,retract,retract,retract,
	extend_right,extend_2x,

	# we're now in the injection receiver. we retract once to inject the sheath, again to inject the initialization signal, then 14 times to back away and give the new machine room
	retract_16x,

	#to stop execution, we're going to extend right and extend, which will intentionally make us collide with the new machine. Then we'll read, which, due to the collision, will not return.
	extend_right,extend,R_read
]

'''
-- ENCODER --
This section contains documentation and python code to actually write the program tape and data tape into the pattern, so that the machine can read them

ENCODING JUMPS/BRANCHES
The compiler will take the written program above and encode the jumps into the program tape via the following:

The destinations for jumps are always 8 bits long, and are 1 bit specifiying forwards/backwards (1 = forawrds, 0=backwards), followed by 7 bits of binary, MSB first, specifying the distance of the jump.

Because all instructions are 4 bits long, and the jump addresses are 8 bits long, all jump destinations are guaranteed to have an alignment of 4 bits.
Therefore, we consider jump distances in "words" of 4 bits, rather than bits themselves. The distance to jump is counted from the end of the address.

Examples:
A jump address of 0 0000001 will jump backwards 1 word.
A jump address of 1 1111111 will jump forwards 127 words
A jump address of 1 1110111 will jump forwards 119 words
A jump address of 0 1100000 will jump backwards 96 words
A jump address of 0 0000000 will jump backwards 0 words, and is a safe no-op. The following instructions will be executed as if no jump took place.


Because the machine doesn't have to read the program tape while executing the jump (and thus doesn't have to wait for the signal to do a round trip), jumps in this machine are very cheap.

Another side effect is that because this machine doesn't jump by counting labels, labels don't increase the size of the program tape in any way and are therefore "free". Instructions can also have multiple labels for added readability.



DATA TAPE ENCODING
Every row begins with a 1. When the program encounters a row that begins with a 0, it will end construction.

Each row contains coded construction instructions:

1   	: write a one
00x		: write 2(x + 1) zeroes, followed by a one, 0 <= x <= 2
01x		: write 2x + 1 zeroes, followed by a one, 0 <= x <= 3
000001yx	: write 6(y + 1) + 2(x + 1) zeroes, followed by a one, 0 <= x <= 2
0100001yx	: write 6(y + 1) + 2(x + 1) + 1 zeroes, followed by a one, 0 <= x <= 2
0000001	: end row

y is represented as a unary sequence of 1's, terminated by a 0.
x is represented as a unary sequence of 0's, terminated by a 1, with a minimum and maximum value. If the encoding goes over the maximum value, the program switches to the chunked "y" mode instead

In english:
The idea here is that we write a 0 to indicate that we're going to write a block of zeroes, then 0 for an even number, and 1 for an odd number. Then, for sizes <= 7 we encode size // 2 as a unary string of zeroes.
For sizes > 7, we instead encode a long sequence of "chunks" of size N, where N (in this case 6) is experimentally determined to produce the smallest result.

Because this encoding scheme is exhaustive of zeroes (ie we will always encode *all* of the zeroes in a contiguous block of zeroes all at once, never just some of them),
we automatically write a one after finishing a block of zeroes, without having to explicitly encode it

Examples [Spaces are purely cosmetic]:
To construct the pattern '1':		1
To construct the pattern '111':		111
To construct the pattern '111111':	111111

To construct the pattern '01':		011						(3 bits to encode 2)
To construct the pattern '001': 	001						(3 bits to encode 3)
To construct the pattern '0001': 	0101					(4 bits to encode 4)
To construct the pattern '00001': 	0001					(4 bits to encode 5)
To construct the pattern '000001':	01001					(5 bits to encode 6)
To construct the pattern '0000001': 00001					(5 bits to encode 7)
To construct the pattern '00000001':010001        			(6 bits to encode 8)

To construct the pattern '000000 001': 		000001  0 1		(8 bits to encode 9)
To construct the pattern '000000 0001': 	0100001 0 1		(9 bits to encode 10)
To construct the pattern '000000 00001': 	000001  0 01	(9 bits to encode 11)
To construct the pattern '000000 000001': 	0100001 0 01	(10 bits to encode 12)
To construct the pattern '000000 0000001': 	000001  0 001	(10 bits to encode 13)
To construct the pattern '000000 000000 01':0100001 0 001	(11 bits to encode 14)

To construct the pattern '000000 000000 001':						000001 1 0 1		(9 bits to encode 15)
To construct the pattern '000000 000000 000000 001':				000001 11 0 1	(10 bits to encode 21)
To construct the pattern '000000 000000 000000 000000 001':			000001 111 0 1	(11 bits to encode 27)
To construct the pattern '000000 000000 000000 000000 000000 001':	000001 1111 0 1	(12 bits to encode 33)


When creating the data tape, each row is scanned from the left edge to the rightmost 1. Rows can be any length, and patterns can have any number of rows.

Based on the encoding scheme above, the longest possible sequence of 0's within the data tape is 6, encountered when we end a row.
To account for this, when copying the data tape, we consider the copy finished when we encounter 7 consecutive 0's.

The left/right position of the data tape, relative to the initial position of the read head, doesn't matter too much.
Before copying (and after copying), the program will move the data head until it encounters the beginning of the data,
allowing some wiggle room in exactly where the data tape is placed, how far we're allowed to overshoot after copying, etc.

To make sure that the program gets copied along with the machine itself, we will first compile the program up avove and write it into the machine. after that we will encode the data tape using the above algorithm
Then we encode both the machine and the program tape into the data tape
'''

import golly as g

program_tape_x = 154
program_tape_y = 105
data_tape_x = 143
data_tape_y = 1
injection_x_left=1
injection_x_right=10
injection_y_center=1

def write_program_bits(s):
	stripe_offset = 6 # the y offset in bits to offset our various striped sections of the program tape
	for (i, c) in enumerate(s):
		if c=='1':
			bit = 1
		else:
			bit = 0

		# The program tape we've been given is one long string of binary, but we want to stripe it into 4 different strings.
		# we'll handle that logic via modular arithmetic here
		x = program_tape_x + i // 4
		y = program_tape_y - (i % 4) * stripe_offset
		g.setcell(x, y, bit)

def write_label_bits(labels):
	x_base = program_tape_x
	y = program_tape_y + 1
	for label in labels:
		g.setcell(x_base + label.address // 4, y, 1)

def write_data_bits(s):
	x = data_tape_x
	for c in s:
		if c=='1':
			bit = 1
		else:
			bit = 0
		g.setcell(x, data_tape_y, bit)
		x += 1

# compile the program and write the serialized bits to the program tape
compiled_program, program_labels = compile_program(program)
write_program_bits(compiled_program)

# Compute the data tape. First, get a bounding rectangle for the machine, plus the encoded program.
rect = g.getrect()
x = rect[0]
y = rect[1]
width = rect[2]
height = rect[3]

chunk_size = 6 # experimentally desetrmined to give both the shortest data tape and fastest replication
assert(chunk_size % 2 == 0) # All of the following encoding code assumes that chunk_size is even

# counts the number of sequential bits with the given value, starting at (row, col) and going right
def count_bits(row, col, value):
	x = col
	total = 0
	while g.getcell(x, row) == value:
		total += 1
		x += 1
	return total

def encode_consecutive_zeros(num_consecutive):
	result = '0'

	# If the number of pixels to draw is odd, get one of the pixels out of the way immediately
	if num_consecutive % 2 == 1:
		num_consecutive -= 1
		result += '1'

	
	# If we have chunk_size or fewer pixels to draw, encode them directly
	if num_consecutive <= chunk_size:
		num_even = num_consecutive // 2
		result += '0' * num_even
		result += '1'

	# we have chunk_size or more pixels to draw
	else:
		result += '0' * (chunk_size // 2 + 1) + '1'

		chunks = num_consecutive // chunk_size

		# if we have a multiple of chunk_size bits to encode, then back off by one chunk so that the final chunk will fall through to the recursive call. that will result in more data to represent this specific case, but much simpler code.
		if num_consecutive % chunk_size == 0:
			chunks -= 1

		# we also get one chunk for free as a part of the decoding process, so we're going to subtract one from this string of ones
		# write a 1 for every actual chunk we're representing
		result += '1' * (chunks - 1)

		# encode the remainder. we know our remainder is an even number with a minimum of 2, so we're going to write a 2 for free.
		remainder = num_consecutive - chunk_size * chunks
		result += encode_consecutive_zeros(remainder - 2)

	return result

def encode_consecutive_ones(num_consecutive):
	# theoretically we could do some sort of run length encoding, but everything i've tried has actually made things slower
	result = '1' * num_consecutive
	return result

def encode_row(row):
	result = ''

	# find te rightmost 1 bit in this row 
	far_right = x+width-1
	while g.getcell(far_right,row)==0:
		far_right -= 1
		if far_right<x:
			g.exit('Empty rows forbidden: '+str(row))

	col = x
	while col <= far_right:
		cell_value = g.getcell(col, row)
		num_consecutive = count_bits(row, col, cell_value)

		if cell_value == 1: # next cell 1?
			col += num_consecutive
			result += encode_consecutive_ones(num_consecutive)
		else:
			col += num_consecutive + 1
			result += encode_consecutive_zeros(num_consecutive)


	result+='0' * (chunk_size // 2 + 3) # end of row
	return result

# now that we have all our subroutines defined, scan through the machine, encode each row, and compose them all to form the data tape
data_tape = ''
for row in reversed(xrange(y, y+height)):

	# each row starts with a "1", so that the program will know when there are no more rows
	data_tape += '1' + encode_row(row)

# write the data tape in reverse. that way we can extend the data arm out to copy it, then retract the data arm back in to decode it
write_data_bits(data_tape[::-1])

#write out markers for each of the program's labels. These are useful for debugging, but aren't actually used by the machine, so we don't include them in the data tape
write_label_bits(program_labels)

# if there are any labels that had no references, warn the user
unused_labels = [label.name for label in program_labels if label.num_references == 0]
if len(unused_labels) > 0:
	g.warn("Unused labels: " + ', '.join(unused_labels))

# tell the user how long the tape was
g.show("Program tape length: %d, Data tape length: %d"%(len(compiled_program)//4, len(data_tape)))

# overwrite the injection receiver with a "seed injector" that will jumpstart the machine
for x in range(injection_x_left, injection_x_right):
	g.setcell(x,injection_y_center-1,2)
	g.setcell(x,injection_y_center  ,1)
	g.setcell(x,injection_y_center+1,2)

g.setcell(injection_x_right  ,injection_y_center,6)
g.setcell(injection_x_right-1,injection_y_center,0)

g.setcell(injection_x_left+2 ,injection_y_center,7)
g.setcell(injection_x_left+1 ,injection_y_center,0)

g.setcell(injection_x_left-1,injection_y_center-1,0)
g.setcell(injection_x_left-1,injection_y_center+1,0)
