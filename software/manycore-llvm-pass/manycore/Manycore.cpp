#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"

using namespace llvm;

#define STRIPE 1 // Needs to match address_space set by library

class FunctionNotFound: public std::exception {
    virtual const char *what() const throw() {
        return "Error: Function not found\n";
    }
} functionNotFoundException;


// Get the size of the overall struct and the offset of the field being accessed.
// Returns -1 if we're not accessing a struct, and the offset otherwise
// The size of the struct is returned in *struct_size if the function returns
// a non-negative value
int get_struct_info(Module &M, Value *ptr_op, unsigned *struct_size) {
    DataLayout layout = DataLayout(&M);
    // Gives struct type
    GEPOperator *gep = dyn_cast<GEPOperator>(ptr_op);
    if (!gep) { return -1;}
    Type *source_type = gep->getSourceElementType();
    while (!isa<StructType>(source_type)) {
        if (isa<SequentialType>(source_type)) {
            source_type = source_type->getSequentialElementType();
        } else {
            return -1;
        }
    }
    StructType *struct_type = cast<StructType>(source_type);
    *struct_size = layout.getTypeAllocSizeInBits(struct_type) / 8;
    // The last operand of GEP gives the index of the field in the struct definition
    unsigned struct_idx = cast<ConstantInt>(gep->getOperand(gep->getNumOperands() - 1))->getSExtValue();
    // Get the offset of the selected field into the struct
    return layout.getStructLayout(struct_type)->getElementOffset(struct_idx);
}


void replace_extern_store(Module &M, StoreInst *op) {
    IRBuilder<> builder(op);
    Function *store_fn;

    std::vector<Value *> args_vector;
    Value *ptr_op = op->getPointerOperand();
    Type *int32_ptr = Type::getInt32PtrTy(M.getContext(),
            dyn_cast<PointerType>(ptr_op->getType())->getAddressSpace());
    Value *ptr_bc = builder.CreatePointerCast(ptr_op, int32_ptr);

    args_vector.push_back(ptr_bc);
    unsigned value_elem_size = op->getValueOperand()->getType()->getPrimitiveSizeInBits() / 8;

    if (value_elem_size == 1) {
        store_fn = M.getFunction("extern_store_char");
    } else if (value_elem_size == 2) {
        store_fn = M.getFunction("extern_store_short");
    } else {
        store_fn = M.getFunction("extern_store_int");
    }
    if (store_fn == NULL) {
       throw functionNotFoundException;
    }
    unsigned struct_size;
    int struct_off = get_struct_info(M, ptr_op, &struct_size);
    if (struct_off < 0) { // Not accessing a struct field
        args_vector.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()),
                    value_elem_size, false));
        args_vector.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()),
                    0, false));
    } else { // Accessing a struct field
        args_vector.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()),
                    struct_size, false));
        args_vector.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()),
                    struct_off, false));
    }
    args_vector.push_back(op->getValueOperand());

    ArrayRef<Value *> args = ArrayRef<Value *>(args_vector);

    // Create the call and replace all uses of the store inst with the call
    Value *new_str = builder.CreateCall(store_fn, args);
    op->replaceAllUsesWith(new_str);

    errs() << "Replace done\n";
    new_str->dump();
}


void replace_extern_load(Module &M, LoadInst *op) {
    IRBuilder<> builder(op);
    Function *load_fn;
    std::vector<Value *> args_vector;

    Value *ptr_op = op->getPointerOperand();
    Type *int32_ptr = Type::getInt32PtrTy(M.getContext(),
            dyn_cast<PointerType>(ptr_op->getType())->getAddressSpace());
    Value *ptr_bc = builder.CreatePointerCast(ptr_op, int32_ptr);

    args_vector.push_back(ptr_bc);
    unsigned value_elem_size = op->getType()->getPrimitiveSizeInBits() / 8;
    if (value_elem_size == 1) {
        load_fn = M.getFunction("extern_load_char");
    } else if (value_elem_size == 2) {
        load_fn = M.getFunction("extern_load_short");
    } else {
        load_fn = M.getFunction("extern_load_int");
    }
    if (load_fn == NULL) {
       throw functionNotFoundException;
    }

    unsigned struct_size;
    int struct_off = get_struct_info(M, ptr_op, &struct_size);
    if (struct_off < 0) { // Not accessing a struct field
        args_vector.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()),
                    value_elem_size, false));
        args_vector.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()),
                    0, false));
    } else { // Accessing a struct field
        args_vector.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()),
                    struct_size, false));
        args_vector.push_back(ConstantInt::get(Type::getInt32Ty(M.getContext()),
                    struct_off, false));
    }

    ArrayRef<Value *> args = ArrayRef<Value *>(args_vector);

    Value *new_ld = builder.CreateCall(load_fn, args);
    op->replaceAllUsesWith(new_ld);
    errs() << "Replace done\n";
    new_ld->dump();
}

void replace_extern_memcpy(Module &M, CallInst *op, bool stripedStore) {
    IRBuilder<> builder(op);
    Function *memcpy_fn;
    if (stripedStore) {
        memcpy_fn = M.getFunction("extern_store_memcpy");
    } else {
        memcpy_fn = M.getFunction("extern_load_memcpy");
    }
    Type *char_ptr = Type::getInt8PtrTy(M.getContext(), 0);
    Type *int_t = Type::getInt32Ty(M.getContext());
    std::vector<Value *> args_vector;

    assert(op->getNumArgOperands() == 4); // dest, src, len, isvolatile

    // Make all arguments pointers to address space 0 so types match runtime fn
    Value *dest_ptr = op->getArgOperand(0);
    Value *src_ptr = op->getArgOperand(1);
    Value *isvol = builder.CreateIntCast(op->getArgOperand(3), int_t, false);
    // Create argument list to pass to memcpy
    args_vector.push_back(dest_ptr);
    args_vector.push_back(src_ptr);
    args_vector.push_back(op->getArgOperand(2));
    ArrayRef<Value *> args = ArrayRef<Value *>(args_vector);

    Value *new_memcpy = builder.CreateCall(memcpy_fn, args);
    op->replaceAllUsesWith(new_memcpy);
    errs() << "Memcpy replace done\n";
    new_memcpy->dump();
}


namespace {
    struct ManycorePass : public ModulePass {
        static char ID;
        ManycorePass() : ModulePass(ID) {}

        int64_t getGlobalVal(Module &M, StringRef name) {
            GlobalVariable *var = M.getGlobalVariable(name);
            if (var) {
                return cast<ConstantInt>(var->getInitializer())->getSExtValue();
            }
            return -1;
        }

        bool isMemcpy(Function *F) {
            if (F != NULL) {
                StringRef fname = F->getName();
                if (fname.startswith(StringRef("llvm.memcpy"))) {
                    return true;
                }
            }
            return false;
        }


        bool shouldReplaceMemcpy(CallInst *op, bool *stripedStore) {
            for (int i = 0; i < op->getNumArgOperands(); i++) {
                if (auto *bc = dyn_cast<BitCastInst>(op->getArgOperand(i))) {
                    if (auto *pt = dyn_cast<PointerType>(bc->getSrcTy())) {
                        if (pt->getAddressSpace() == STRIPE) {
                            *stripedStore = (i == 0);
                            return true;
                        }
                    }
                    if (auto *pt = dyn_cast<PointerType>(bc->getDestTy())) {
                        if (pt->getAddressSpace() == STRIPE) {
                            *stripedStore = (i == 0);
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        class AddressSpaceException: public std::exception {
            virtual const char *what() const throw() {
                return "Invalid Address Space Encountered!\n";
            }
        } addressSpaceException;

        bool runOnModule(Module &M) override {
            // If the function doesn't exist, it means that this c file didn't
            // include the striping functions.
            if (!M.getFunction("extern_load_int")) {
                return false;
            }
            std::vector<GlobalVariable *> globals_to_resize;
            std::vector<Instruction *> insts_to_remove;
            for (auto &G : M.globals()) {
                Type *g_type = G.getType();
                // If global variable is an array in address space 1
                if (isa<PointerType>(g_type) && g_type->getPointerAddressSpace() > 0) {
                    globals_to_resize.push_back(&G);
                }
            }
            for (auto G: globals_to_resize) {
                // We set alignment so that index 0 of an array is always on
                // core 0. Additionally, this has the effect of the start of
                // a striped array being word-aligned on individual cores,
                // which is arguably more important.
                //
                // bsg_group_size is passed via the command line
                G->setAlignment(4 * bsg_group_size);
                G->setSection(".striped.data");
            }

            for (auto &F : M) {
                for (auto &B : F) {
                    for (auto &I : B) {
                        if (auto* op = dyn_cast<StoreInst>(&I)) {
                            if (op->getPointerAddressSpace() > 0) {
                                op->dump();
                                if (op->getPointerAddressSpace() == STRIPE) {
                                    replace_extern_store(M, op);
                                    insts_to_remove.push_back(op);
                                } else {
                                    throw addressSpaceException;
                                }
                                errs() << "\n";
                            }
                        } else if (auto* op = dyn_cast<LoadInst>(&I)) {
                            if (op->getPointerAddressSpace() > 0) {
                                op->dump();
                                if (op->getPointerAddressSpace() == STRIPE) {
                                    replace_extern_load(M, op);
                                    insts_to_remove.push_back(op);
                                } else {
                                    throw addressSpaceException;
                                }
                                errs() << "\n";
                            }
                        } else if (auto* op = dyn_cast<CallInst>(&I)) {
                            Function *F = op->getCalledFunction();
                            if (!isMemcpy(F)) { continue;}
                            op->dump();
                            bool stripedStore;
                            if (shouldReplaceMemcpy(op, &stripedStore)) {
                                replace_extern_memcpy(M, op, stripedStore);
                                insts_to_remove.push_back(op);
                            }
                            errs() << "\n";
                        }
                    }
                }
            }
            // Can't erase while iterating, so we do it here
            for (auto I : insts_to_remove) {
                I->eraseFromParent();
            }

            std::vector<Function *> funcs_to_internalize;
            funcs_to_internalize.push_back(M.getFunction("extern_store_char"));
            funcs_to_internalize.push_back(M.getFunction("extern_store_short"));
            funcs_to_internalize.push_back(M.getFunction("extern_store_int"));
            funcs_to_internalize.push_back(M.getFunction("extern_load_char"));
            funcs_to_internalize.push_back(M.getFunction("extern_load_short"));
            funcs_to_internalize.push_back(M.getFunction("extern_load_int"));
            for (auto F : funcs_to_internalize) {
                Attribute attr = Attribute::get(M.getContext(), "static", "true");
                F->addAttribute(0, attr);
            }

            return true;
        }
    };
}

char ManycorePass::ID = 0;
static RegisterPass<ManycorePass> X("manycore", "Manycore addressing Pass",
        false /* Doesn't only look at CFG */,
        false /* Isn't an Analysis Pass -- modifies code */);