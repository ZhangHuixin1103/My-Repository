#include "assert.H"
#include "exceptions.H"
#include "console.H"
#include "paging_low.H"
#include "page_table.H"

#define PAGE_DIRECTORY_FRAME_SIZE 1

#define PAGE_PRESENT 1
#define PAGE_WRITE 2
#define PAGE_LEVEL 4

#define PD_SHIFT 22
#define PT_SHIFT 12

#define PDE_MASK 0xFFFFF000
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
    page_directory = (unsigned long *)(kernel_mem_pool->get_frames(PAGE_DIRECTORY_FRAME_SIZE) * PAGE_SIZE);

    unsigned long mask_addr = 0;
    unsigned long *dir_map_page_table = (unsigned long *)(kernel_mem_pool->get_frames(PAGE_DIRECTORY_FRAME_SIZE) * PAGE_SIZE);
    unsigned long shared_frames = (PageTable::shared_size / PAGE_SIZE);

    for (int i = 0; i < shared_frames; i++)
    {
        dir_map_page_table[i] = mask_addr | PAGE_WRITE | PAGE_PRESENT;
        mask_addr += PAGE_SIZE;
    }

    page_directory[0] = (unsigned long)dir_map_page_table | PAGE_WRITE | PAGE_PRESENT;

    mask_addr = 0;

    for (int i = 1; i < shared_frames; i++)
    {
        page_directory[i] = mask_addr | PAGE_WRITE;
    }

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
    unsigned long *cur_page_directory = (unsigned long *)read_cr3();

    unsigned long page_addr = read_cr2();
    unsigned long pd_addr = page_addr >> PD_SHIFT;
    unsigned long pt_addr = page_addr >> PT_SHIFT;

    unsigned long *page_table = NULL;
    unsigned long error_code = _r->err_code;

    unsigned long mask_addr = 0;

    if ((error_code & PAGE_PRESENT) == 0)
    {
        if ((cur_page_directory[pd_addr] & PAGE_PRESENT) == 1)
        {
            page_table = (unsigned long *)(cur_page_directory[pd_addr] & PDE_MASK);
            page_table[pt_addr & PT_MASK] = (PageTable::process_mem_pool->get_frames(PAGE_DIRECTORY_FRAME_SIZE) * PAGE_SIZE) | PAGE_WRITE | PAGE_PRESENT;
        }
        else
        {
            cur_page_directory[pd_addr] = (unsigned long)((kernel_mem_pool->get_frames(PAGE_DIRECTORY_FRAME_SIZE) * PAGE_SIZE) | PAGE_WRITE | PAGE_PRESENT);

            page_table = (unsigned long *)(cur_page_directory[pd_addr] & PDE_MASK);

            for (int i = 0; i < 1024; i++)
            {
                page_table[i] = mask_addr | PAGE_LEVEL;
            }

            page_table[pt_addr & PT_MASK] = (PageTable::process_mem_pool->get_frames(PAGE_DIRECTORY_FRAME_SIZE) * PAGE_SIZE) | PAGE_WRITE | PAGE_PRESENT;
        }
    }

    Console::puts("handled page fault\n");
}
