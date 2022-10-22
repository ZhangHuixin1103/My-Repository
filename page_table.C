#include "assert.H"
#include "exceptions.H"
#include "console.H"
#include "paging_low.H"
#include "page_table.H"

#define PAGE_DIR_FRAME_SIZE 1

#define PAGE_PRESENT 1
#define PAGE_WRITE 2
#define PAGE_LEVEL 4

#define PD_SHIFT 22
#define PT_SHIFT 12
#define PD_MASK 0xFFFFF000
#define PT_MASK 0x3FF

PageTable *PageTable::current_page_table = NULL;
unsigned int PageTable::paging_enabled = 0;
ContFramePool *PageTable::kernel_mem_pool = NULL;
ContFramePool *PageTable::process_mem_pool = NULL;
unsigned long PageTable::shared_size = 0;

void PageTable::init_paging(ContFramePool *_kernel_mem_pool,
                            ContFramePool *_process_mem_pool,
                            const unsigned long _shared_size)
{
    PageTable::kernel_mem_pool = _kernel_mem_pool;
    PageTable::process_mem_pool = _process_mem_pool;
    PageTable::shared_size = _shared_size;

    Console::puts("Initialized Paging System\n");
}

PageTable::PageTable()
{
    page_directory = (unsigned long *)(process_mem_pool->get_frames(PAGE_DIR_FRAME_SIZE) * PAGE_SIZE);

    unsigned long mask_address = 0;
    unsigned long *dir_map_page_table = (unsigned long *)(process_mem_pool->get_frames(PAGE_DIR_FRAME_SIZE) * PAGE_SIZE);
    unsigned long shared_frames = (PageTable::shared_size / PAGE_SIZE);

    for (int i = 0; i < shared_frames; i++)
    {
        dir_map_page_table[i] = mask_address | PAGE_WRITE | PAGE_PRESENT;
        mask_address += PAGE_SIZE;
    }

    page_directory[0] = (unsigned long)dir_map_page_table | PAGE_WRITE | PAGE_PRESENT;

    mask_address = 0;

    for (int i = 1; i < shared_frames; i++)
    {
        page_directory[i] = mask_address | PAGE_WRITE;
    }

    page_directory[shared_frames - 1] = (unsigned long)(page_directory) | PAGE_WRITE | PAGE_PRESENT;

    for (int i = 0; i < VM_POOL_SIZE; i++)
    {
        reg_vm_pool[i] = NULL;
    }
    vm_pool_num = 0;

    Console::puts("Constructed Page Table object\n");
}

void PageTable::load()
{
    current_page_table = this;
    write_cr3((unsigned long)page_directory);

    Console::puts("Loaded page table\n");
}

void PageTable::enable_paging()
{
    paging_enabled = 1;
    write_cr0(read_cr0() | 0x80000000);

    Console::puts("Enabled paging\n");
}

void PageTable::handle_fault(REGS *_r)
{
    unsigned long page_address = read_cr2();
    unsigned long pd_address = page_address >> PD_SHIFT;
    unsigned long pt_address = page_address >> PT_SHIFT;

    unsigned long *page_table = NULL;
    unsigned long error_code = _r->err_code;

    unsigned long mask_address = 0;

    unsigned long *cur_page_directory = (unsigned long *)0xFFFFF000;

    if ((error_code & PAGE_PRESENT) == 0)
    {
        int index = -1;
        VMPool **vm_pool = current_page_table->reg_vm_pool;
        for (int i = 0; i < current_page_table->vm_pool_num; i++)
        {
            if (vm_pool[i] != NULL)
            {
                if (vm_pool[i]->is_legitimate(page_address))
                {
                    index = i;
                    break;
                }
            }
        }
        assert(!(index < 0));

        if ((cur_page_directory[pd_address] & PAGE_PRESENT) == 1)
        {
            page_table = (unsigned long *)(0xFFC00000 | (pd_address << PT_SHIFT));
            page_table[pt_address & PT_MASK] = (PageTable::process_mem_pool->get_frames(PAGE_DIR_FRAME_SIZE) * PAGE_SIZE) | PAGE_WRITE | PAGE_PRESENT;
        }
        else
        {
            cur_page_directory[pd_address] = (unsigned long)((process_mem_pool->get_frames(PAGE_DIR_FRAME_SIZE) * PAGE_SIZE) | PAGE_WRITE | PAGE_PRESENT);

            page_table = (unsigned long *)(0xFFC00000 | (pd_address << PT_SHIFT));

            for (int i = 0; i < 1024; i++)
            {
                page_table[i] = mask_address | PAGE_LEVEL;
            }

            page_table[pt_address & PT_MASK] = (PageTable::process_mem_pool->get_frames(PAGE_DIR_FRAME_SIZE) * PAGE_SIZE) | PAGE_WRITE | PAGE_PRESENT;
        }
    }

    Console::puts("handled page fault\n");
}

void PageTable::register_pool(VMPool *_vm_pool)
{
    if (vm_pool_num < VM_POOL_SIZE)
    {
        reg_vm_pool[vm_pool_num++] = _vm_pool;
        Console::puts("registered VM pool\n");
    }
    else
    {
        Console::puts("VM POOL cannot register");
    }
}

void PageTable::free_page(unsigned long _page_no)
{
    unsigned long pd_address = _page_no >> PD_SHIFT;
    unsigned long pt_address = _page_no >> PT_SHIFT;

    unsigned long *page_table = (unsigned long *)(0xFFC00000 | (pd_address << PT_SHIFT));

    unsigned long frame_num = page_table[pt_address & PT_MASK] / (Machine::PAGE_SIZE);
    process_mem_pool->release_frames(frame_num);

    page_table[pt_address & PT_MASK] = 0 | PAGE_WRITE;

    Console::puts("freed page\n");
}
