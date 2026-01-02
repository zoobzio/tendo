package tendo

// Tape records tensor operations during forward pass for backward computation.
// Each operation appends an entry containing saved tensors needed for its backward pass.
type Tape struct {
	entries []TapeEntry
}

// TapeEntry records one operation's saved tensors for backward pass.
type TapeEntry struct {
	Saved  map[string]*Tensor // tensors saved for backward (e.g., "input", "output")
	OpName string             // operation name (e.g., "relu", "matmul")
}

// NewTape creates an empty tape for recording operations.
func NewTape() *Tape {
	return &Tape{
		entries: make([]TapeEntry, 0),
	}
}

// Record adds an operation entry to the tape.
func (t *Tape) Record(opName string, saved map[string]*Tensor) {
	t.entries = append(t.entries, TapeEntry{
		OpName: opName,
		Saved:  saved,
	})
}

// Len returns the number of recorded operations.
func (t *Tape) Len() int {
	return len(t.entries)
}

// Entry returns the entry at index i.
// Panics if i is out of bounds.
func (t *Tape) Entry(i int) TapeEntry {
	return t.entries[i]
}

// Entries returns all tape entries.
func (t *Tape) Entries() []TapeEntry {
	return t.entries
}

// Clear removes all entries from the tape.
func (t *Tape) Clear() {
	t.entries = t.entries[:0]
}
